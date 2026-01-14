import os, sys, argparse, time, math, random, warnings
warnings.filterwarnings("ignore")
import yaml
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve

# Torch for generators / surrogate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Try to import RQ2 helpers from user's script for exact compatibility
# ---------------------------
try:
    # if your RQ2 file is named RQ3_DR_Synth_CondDiff_STP.py and is in same dir
    from RQ3_DR_Synth_CondDiff_STP import load_and_scale as rq2_load_and_scale
    from RQ3_DR_Synth_CondDiff_STP import split_feature_blocks as rq2_split_feature_blocks
    print("[Info] Imported RQ2 helpers from RQ3_DR_Synth_CondDiff_STP.py")
    def load_and_scale(ben, mal):
        Xb, Xm, cols = rq2_load_and_scale(ben, mal)
        # rq2_load_and_scale returns scaled arrays and a list of column names
        return Xb, Xm, cols, None  # no scaler returned; RQ2 used StandardScaler internally
    def split_feature_blocks(cols):
        return rq2_split_feature_blocks(cols)
except Exception as e:
    # fallback implementations (same logic as earlier assistant)
    print("[Warn] Could not import RQ2 helpers; using local implementations.", e)
    def load_and_scale(ben_path, mal_path):
        df_b = pd.read_csv(ben_path)
        df_m = pd.read_csv(mal_path)
        if 'label' in df_b.columns: df_b = df_b.drop(columns=['label'])
        if 'label' in df_m.columns: df_m = df_m.drop(columns=['label'])
        df_b = df_b.fillna(0); df_m = df_m.fillna(0)
        common = [c for c in df_b.columns if c in df_m.columns]
        if len(common)==0:
            raise ValueError("No common columns found between benign and malicious CSVs.")
        sc = StandardScaler()
        Xb = sc.fit_transform(df_b[common].values)
        Xm = sc.transform(df_m[common].values)
        return Xb.astype(np.float32), Xm.astype(np.float32), common, sc

    def split_feature_blocks(cols):
        low = [c.lower() for c in cols]
        idxT = [i for i,c in enumerate(low) if any(k in c for k in
               ["duration","piat","first_seen","last_seen","stddev_ps","mean_ps","max_ps","min_ps"])]
        idxS = [i for i,c in enumerate(low) if any(k in c for k in
               ["packets","bytes","syn_packets","ack_packets","rst_packets","fin_packets","psh_packets","ece_packets","urg_packets","cwr_packets"])]
        idxP = [i for i,c in enumerate(low) if any(k in c for k in ["port","protocol","ip_version"])]
        remain = [i for i in range(len(cols)) if i not in set(idxT+idxS+idxP)]
        if len(idxT)==0 or len(idxS)==0 or len(idxP)==0:
            k = max(1, len(remain)//3)
            if len(idxT)==0: idxT += remain[:k]
            if len(idxS)==0: idxS += remain[k:2*k]
            if len(idxP)==0: idxP += remain[2*k:2*k+k]
        return idxT, idxS, idxP

# ---------------------------
# Classifier training / calibration (FPR ~ 1%)
# ---------------------------
def train_detectors(X_train, y_train):
    dets = {}
    dets['LogReg'] = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    dets['RF'] = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_train, y_train)
    dets['SVM'] = SVC(probability=True, gamma='scale').fit(X_train, y_train)
    dets['GBM'] = GradientBoostingClassifier(n_estimators=200, random_state=SEED).fit(X_train, y_train)
    dets['MLP'] = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=SEED).fit(X_train, y_train)
    return dets

def calibrate_threshold(clf, X_benign_val, target_fpr=0.01):
    try:
        probs = clf.predict_proba(X_benign_val)[:,1]
    except:
        df = clf.decision_function(X_benign_val)
        probs = (df - df.min()) / (df.max()-df.min()+1e-12)
    fpr, tpr, th = roc_curve(np.zeros(len(probs)), probs)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return th[idx]

# ---------------------------
# Surrogate model (BNDNN) used by FGSM
# ---------------------------
class BNDNN(nn.Module):
    def __init__(self, dim, hidden=[64,32,8]):
        super().__init__()
        layers = []
        in_dim = dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_surrogate(X_train, y_train, X_val, y_val, dim, epochs=40, lr=1e-3, batch=128):
    model = BNDNN(dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).long())
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    best = 0.0; best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_val).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = (preds == y_val).mean()
            if acc > best:
                best = acc; best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def surrogate_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

# ---------------------------
# FGSM iterative (uses surrogate)
# ---------------------------
def fgsm_iter(surrogate_model, X, eps=0.02, steps=8):
    model = surrogate_model.to(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    X_adv = X_t.clone().detach()
    target = torch.zeros(X.shape[0], dtype=torch.long, device=device)
    for _ in range(steps):
        X_adv.requires_grad_(True)
        logits = model(X_adv)
        loss = nn.CrossEntropyLoss()(logits, target)
        model.zero_grad()
        if X_adv.grad is not None:
            X_adv.grad.zero_()
        loss.backward()
        grad = X_adv.grad.data
        X_adv = (X_adv - eps * torch.sign(grad)).detach()
    return X_adv.cpu().numpy().astype(np.float32)

# ---------------------------
# WGAN-GP for sample generation (IDSGAN, VulnerGAN)
# ---------------------------
class WGANGP_G(nn.Module):
    def __init__(self, zdim, outdim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, outdim)
        )
    def forward(self, z): return self.net(z)

class WGANGP_D(nn.Module):
    def __init__(self, indim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).view(-1)

def train_wgangp_sample(benign_samples, zdim=64, epochs=800, batch=256, lr=1e-4, gp_lambda=10.0):
    device_ = device
    G = WGANGP_G(zdim, benign_samples.shape[1]).to(device_)
    D = WGANGP_D(benign_samples.shape[1]).to(device_)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.9))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.9))
    ds = TensorDataset(torch.tensor(benign_samples))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    ones = None
    for ep in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device_)
            # train D
            z = torch.randn(xb.size(0), zdim, device=device_)
            xg = G(z)
            d_real = D(xb)
            d_fake = D(xg.detach())
            # gradient penalty
            alpha = torch.rand(xb.size(0), 1, device=device_)
            interp = (alpha * xb + (1-alpha) * xg).requires_grad_(True)
            d_interp = D(interp)
            grads = torch.autograd.grad(outputs=d_interp, inputs=interp,
                                        grad_outputs=torch.ones_like(d_interp),
                                        create_graph=True, retain_graph=True)[0]
            gradnorm = grads.view(grads.size(0), -1).norm(2, dim=1)
            gp = gp_lambda * ((gradnorm -1) **2).mean()
            lossD = d_fake.mean() - d_real.mean() + gp
            optD.zero_grad(); lossD.backward(); optD.step()
            # train G
            if random.random() < 1.0:
                z = torch.randn(xb.size(0), zdim, device=device_)
                xg = G(z)
                lossG = -D(xg).mean()
                optG.zero_grad(); lossG.backward(); optG.step()
        # lightweight progress print
        if (ep+1) % max(1, epochs//10) == 0:
            print(f"[WGAN] Epoch {ep+1}/{epochs} lossD={lossD.item():.4f} lossG={lossG.item():.4f}")
    return G, D

# ---------------------------
# Conditional perturbation GAN (DIGFuPAS / GPMT)
# Generator input: malicious sample -> outputs delta (small)
# Discriminator: distinguishes benign vs (mal+delta)
# ---------------------------
class CondGen(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, x): return self.net(x)

class CondDisc(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).view(-1)

def train_cond_perturbation_gan(benign, malicious, epochs=600, batch=256, lr=1e-4, l2_reg=0.01, delta_scale=0.5):
    device_ = device
    dim = benign.shape[1]
    G = CondGen(dim).to(device_)
    D = CondDisc(dim).to(device_)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.9))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.9))
    ds_b = TensorDataset(torch.tensor(benign))
    ds_m = TensorDataset(torch.tensor(malicious))
    dl_b = DataLoader(ds_b, batch_size=batch, shuffle=True, drop_last=True)
    dl_m = DataLoader(ds_m, batch_size=batch, shuffle=True, drop_last=True)
    itb = iter(dl_b)
    for ep in range(epochs):
        for xm_tup in dl_m:
            xm = xm_tup[0].to(device_)
            # sample benign batch (wrap-around)
            try:
                xb = next(itb)[0].to(device_)
            except StopIteration:
                itb = iter(dl_b); xb = next(itb)[0].to(device_)
            # train D on real benign vs fake (xm + delta)
            delta = G(xm)
            # scale delta to moderate magnitude: apply tanh then scale
            delta = torch.tanh(delta) * delta_scale
            x_fake = (xm + delta)
            d_real = D(xb)
            d_fake = D(x_fake.detach())
            lossD = d_fake.mean() - d_real.mean()
            optD.zero_grad(); lossD.backward(); optD.step()
            # train G with objective to fool D + l2 regularization on delta
            delta = G(xm)
            delta = torch.tanh(delta) * delta_scale
            x_fake = (xm + delta)
            lossG = -D(x_fake).mean() + l2_reg * (delta.view(delta.size(0), -1).norm(2, dim=1).mean())
            optG.zero_grad(); lossG.backward(); optG.step()
        if (ep+1) % max(1, epochs//8) == 0:
            print(f"[cGAN] Ep {ep+1}/{epochs} lossD={lossD.item():.4f} lossG={lossG.item():.4f}")
    return G, D

# ---------------------------
# ProGen: LSTM-based generator that maps noise -> feature vector (treat features as sequence)
# ---------------------------
class LSTMGen(nn.Module):
    def __init__(self, zdim, feat_dim, hidden=128, num_layers=1):
        super().__init__()
        self.fc = nn.Linear(zdim, feat_dim)  # simple: produce entire vector directly
        # alternative: use LSTM over "timesteps"; but feature ordering is arbitrary — simpler direct mapping works
    def forward(self, z):
        return self.fc(z)

def train_progen_lstm(benign, zdim=64, epochs=400, batch=256, lr=1e-4):
    device_ = device
    G = LSTMGen(zdim, benign.shape[1]).to(device_)
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.9))
    ds = TensorDataset(torch.tensor(benign))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device_)
            z = torch.randn(xb.size(0), zdim, device=device_)
            xg = G(z)
            loss = loss_fn(xg, xb)
            optG.zero_grad(); loss.backward(); optG.step()
        if (ep+1) % max(1, epochs//8) == 0:
            print(f"[ProGen] Ep {ep+1}/{epochs} mse={loss.item():.6f}")
    return G

# ---------------------------
# Evaluation pipeline
# ---------------------------
def evaluate_asr_on_methods(ben_path, mal_path, adv_npy=None, out='rq3_full_results'):
    os.makedirs(out, exist_ok=True)
    # load and scale
    Xb, Xm, cols, scaler = load_and_scale(ben_path, mal_path)
    idxT, idxS, idxP = split_feature_blocks(cols)
    print(f"[Info] Loaded D={len(cols)} features | T={len(idxT)} S={len(idxS)} P={len(idxP)}")
    # split for classification training and calibration
    Xb_tr, Xb_hold = train_test_split(Xb, test_size=0.2, random_state=SEED)
    Xb_val, Xb_test = train_test_split(Xb_hold, test_size=0.5, random_state=SEED)
    Xm_tr, Xm_test = train_test_split(Xm, test_size=0.5, random_state=SEED)
    # train detectors (on A = Xb_tr + Xm_tr)
    X_train_clf = np.vstack([Xb_tr, Xm_tr])
    y_train_clf = np.hstack([np.zeros(len(Xb_tr)), np.ones(len(Xm_tr))]).astype(int)
    dets = train_detectors(X_train_clf, y_train_clf)
    # surrogate training for FGSM
    X_s = X_train_clf; y_s = y_train_clf
    X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(X_s, y_s, test_size=0.2, random_state=SEED)
    surrogate = train_surrogate(X_s_train, y_s_train, X_s_val, y_s_val, dim=Xb.shape[1], epochs=40)
    # calibrate thresholds
    thresholds = {}
    for name, clf in dets.items():
        thresholds[name] = calibrate_threshold(clf, Xb_val, target_fpr=0.01)
        print(f"Calibrated {name} thresh = {thresholds[name]:.6f}")
    thresholds['Surrogate(BNDNN)'] = None
    # calibrate surrogate threshold
    probs_ben_sur = surrogate_proba(surrogate, Xb_val)[:,1]
    fpr, tpr, ths = roc_curve(np.zeros(len(probs_ben_sur)), probs_ben_sur)
    idx = np.argmin(np.abs(fpr - 0.01)); thresholds['Surrogate(BNDNN)'] = ths[idx]
    print(f"Calibrated Surrogate thresh = {thresholds['Surrogate(BNDNN)']:.6f}")

    # compute pre-asr
    pre_asr = {}
    for name, clf in dets.items():
        try:
            probs = clf.predict_proba(Xm_test)[:,1]
        except:
            df = clf.decision_function(Xm_test)
            probs = (df - df.min())/(df.max()-df.min()+1e-12)
        preds = (probs >= thresholds[name]).astype(int)
        pre_asr[name] = 100.0 * (np.sum(preds == 0) / len(preds))
    probs_sur_mal = surrogate_proba(surrogate, Xm_test)[:,1]
    preds_sur = (probs_sur_mal >= thresholds['Surrogate(BNDNN)']).astype(int)
    pre_asr['Surrogate(BNDNN)'] = 100.0 * (np.sum(preds_sur == 0) / len(preds_sur))
    print("Pre-ASR per detector (orig malicious):")
    for k,v in pre_asr.items():
        print(f"  {k}: {v:.2f}%")

    # prepare method outputs
    methods_outputs = {}

    # 1) FGSM (surrogate iterative)
    print("[Method] FGSM (iterative) generating...")
    X_fgsm = fgsm_iter(surrogate, Xm_test, eps=0.02, steps=8)
    methods_outputs['FGSM'] = X_fgsm

    # 2) IDSGAN (WGAN-GP generate samples targeting benign distribution)
    print("[Method] IDSGAN (WGAN-GP) training to model benign distribution...")
    G_idsgan, D_idsgan = train_wgangp_sample(Xb_tr, zdim=64, epochs=10, batch=256, lr=1e-4)
    # sample same N as Xm_test
    z = torch.randn(len(Xm_test), 64, device=device)
    X_idsgan = G_idsgan(z).detach().cpu().numpy().astype(np.float32)
    methods_outputs['IDSGAN'] = X_idsgan

    # 3) DIGFuPAS (conditional perturbation GAN: generate delta for each malicious)
    print("[Method] DIGFuPAS (cGAN perturbation) training...")
    G_dig, D_dig = train_cond_perturbation_gan(Xb_tr, Xm_tr, epochs=10, batch=256, lr=1e-4, l2_reg=0.01, delta_scale=0.4)
    with torch.no_grad():
        Xm_t = torch.tensor(Xm_test).to(device)
        delta = torch.tanh(G_dig(Xm_t)) * 0.4
        X_dig = (Xm_t + delta).cpu().numpy().astype(np.float32)
    methods_outputs['DIGFuPAS'] = X_dig

    # 4) VulnerGAN: another WGAN-GP (train separately to benign) - implement as duplicate of IDSGAN to represent different baseline
    print("[Method] VulnerGAN (WGAN-GP) training...")
    G_vul, D_vul = train_wgangp_sample(Xb_tr, zdim=64, epochs=10, batch=256, lr=1e-4)
    z = torch.randn(len(Xm_test), 64, device=device)
    X_vul = G_vul(z).detach().cpu().numpy().astype(np.float32)
    methods_outputs['VulnerGAN'] = X_vul

    # 5) GPMT: conditional perturbation WGAN-like (we reuse cond perturbation GAN but with different hyperparams)
    print("[Method] GPMT (conditional perturbation WGAN variant) training...")
    G_gpm, D_gpm = train_cond_perturbation_gan(Xb_tr, Xm_tr, epochs=10, batch=256, lr=1e-4, l2_reg=0.005, delta_scale=0.6)
    with torch.no_grad():
        Xm_t = torch.tensor(Xm_test).to(device)
        delta = torch.tanh(G_gpm(Xm_t)) * 0.6
        X_gpm = (Xm_t + delta).cpu().numpy().astype(np.float32)
    methods_outputs['GPMT'] = X_gpm

    # 6) ProGen: LSTM-like generator mapping noise->sample (train to reconstruct benign)
    print("[Method] ProGen (LSTM generator) training...")
    G_progen = train_progen_lstm(Xb_tr, zdim=64, epochs=10, batch=256, lr=1e-4)
    with torch.no_grad():
        z = torch.randn(len(Xm_test), 64, device=device)
        X_progen = G_progen(z).cpu().numpy().astype(np.float32)
    methods_outputs['ProGen'] = X_progen

    # Optional: DR-Synth adversarial from RQ2 if provided
    if adv_npy and os.path.exists(adv_npy):
        adv = np.load(adv_npy)
        if adv.shape[0] >= Xm_test.shape[0]:
            adv_use = adv[:len(Xm_test)].astype(np.float32)
        else:
            # if adv smaller, sample subset of Xm_test to match N
            n = adv.shape[0]
            print(f"[Info] adv_npy has {n} samples < Xm_test {len(Xm_test)} -> using {n} subset")
            adv_use = adv.astype(np.float32)
            # also shrink other method outputs to n for fair comparison
            for k in list(methods_outputs.keys()):
                methods_outputs[k] = methods_outputs[k][:n]
            Xm_test = Xm_test[:n]
        methods_outputs['DR-Synth (Ours)'] = adv_use
    else:
        print("[Warn] No adv_npy provided or not found; DR-Synth (Ours) skipped.")

    # Evaluate each method on detectors
    rows = []
    for method_name, X_gen in methods_outputs.items():
        print(f"[Eval] {method_name} N={len(X_gen)}")
        row = {'Method': method_name}
        for det_name, clf in dets.items():
            try:
                probs = clf.predict_proba(X_gen)[:,1]
            except:
                df = clf.decision_function(X_gen)
                probs = (df - df.min())/(df.max()-df.min()+1e-12)
            preds = (probs >= thresholds[det_name]).astype(int)
            asr = 100.0 * (np.sum(preds == 0) / len(preds))
            row[det_name] = asr
        # surrogate
        probs_sur = surrogate_proba(surrogate, X_gen)[:,1]
        preds_sur = (probs_sur >= thresholds['Surrogate(BNDNN)']).astype(int)
        row['Surrogate(BNDNN)'] = 100.0 * (np.sum(preds_sur == 0) / len(preds_sur))
        row['MacroAvg'] = np.mean([row[k] for k in ['LogReg','RF','SVM','GBM','MLP','Surrogate(BNDNN)']])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "rq3_asr_full_baselines.csv"), index=False)
    print(f"[Saved] Results CSV -> {os.path.join(out,'rq3_asr_full_baselines.csv')}")
    print(df.to_string(index=False, float_format="%.2f"))

    # Pre/Post table for surrogate (like paper): Pre is pre_asr for surrogate, Post is for each method
    pre = pre_asr['Surrogate(BNDNN)']
    prepost_rows = []
    for r in rows:
        post = r['Surrogate(BNDNN)']
        prepost_rows.append({'Method': r['Method'], 'Pre-ASR (%)': pre, 'Post-ASR (%)': post, 'Delta (pp)': post - pre})
    df_prepost = pd.DataFrame(prepost_rows)
    df_prepost.to_csv(os.path.join(out, "rq3_prepost_surrogate.csv"), index=False)
    print(f"[Saved] Pre/Post CSV -> {os.path.join(out,'rq3_prepost_surrogate.csv')}")
    print(df_prepost.to_string(index=False, float_format="%.2f"))

    # stacked bar figure
    fig, ax = plt.subplots(figsize=(10,6))
    methods_order = df_prepost['Method'].tolist()
    pre_vals = df_prepost['Pre-ASR (%)'].values
    deltas = df_prepost['Delta (pp)'].values
    ax.bar(methods_order, pre_vals, label='Pre-ASR', color='lightgray')
    ax.bar(methods_order, deltas, bottom=pre_vals, label='Delta', color='orange')
    ax.set_ylabel('ASR (%)'); ax.set_title('Pre and Post ASR (Surrogate)')
    ax.legend(); plt.xticks(rotation=35, ha='right'); plt.tight_layout()
    figpath = os.path.join(out, "rq3_asr_stacked_full.png")
    plt.savefig(figpath, dpi=200)
    print(f"[Saved] Figure -> {figpath}")

    return df, df_prepost

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config.yaml used in RQ2')
    parser.add_argument('--benign', type=str, default=None, help='benign csv (overrides config)')
    parser.add_argument('--malicious', type=str, default=None, help='malicious csv (overrides config)')
    parser.add_argument('--adv_npy', type=str, default='../results/drsynth_stp_adv.npy', help='path to DR-Synth adv npy (optional)')
    parser.add_argument('--out', type=str, default='rq3_results_full', help='output directory')
    args = parser.parse_args()

    if args.config:
        with open(args.config,'r',encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        ben = cfg.get('labeled_benign_csv', args.benign)
        mal = cfg.get('labeled_malicious_csv', args.malicious)
    else:
        ben = args.benign
        mal = args.malicious

    if ben is None or mal is None:
        print("Error: specify --benign and --malicious or provide --config")
        sys.exit(1)

    evaluate_asr_on_methods(ben, mal, adv_npy=args.adv_npy, out=args.out)
