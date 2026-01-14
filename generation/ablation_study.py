import os, yaml, time, numpy as np
from RQ3_DR_Synth_CondDiff_STP import (
    load_and_scale, split_feature_blocks, calculate_metrics,
    train_cond_diffusion, sample_adv_from_mal, light_stp_loss
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# ✅ 封装一轮训练-采样-评估
# -------------------------------
def run_experiment(cfg_path, desc, **overrides):
    """运行一轮实验并返回指标 dict"""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    X_ben, X_mal, cols = load_and_scale(cfg['labeled_benign_csv'], cfg['labeled_malicious_csv'])
    idxT, idxS, idxP = split_feature_blocks(cols)

    from RQ3_DR_Synth_CondDiff_STP import (
        EpsModel, MalEncoder, DDPM, light_stp_loss
    )
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from itertools import cycle

    # 读取超参默认值
    EPOCHS = overrides.get("EPOCHS", 40)
    LR = overrides.get("LR", 1e-3)
    TIMESTEPS = overrides.get("TIMESTEPS", 400)
    LAMBDA_HSIC = overrides.get("LAMBDA_HSIC", 0.05)
    USE_STP_LOSS = overrides.get("USE_STP_LOSS", True)
    STP_PAIR_MODE = overrides.get("STP_PAIR_MODE", "ALL")
    USE_COND = overrides.get("USE_COND", True)

    print(f"\n🧪 [{desc}] | λ={LAMBDA_HSIC} | STP={STP_PAIR_MODE} | Cond={USE_COND} | T={TIMESTEPS}")

    # ========== 定义网络 ==========
    LATENT_T = 128
    dim = X_ben.shape[1]

    class EpsModelAbl(nn.Module):
        def __init__(self, dim_x, dim_cond, hidden=256, use_cond=True):
            super().__init__()
            self.use_cond = use_cond
            in_dim = dim_x + 1 + (dim_cond if use_cond else 0)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, dim_x)
            )
        def forward(self, x_t, t, cond=None):
            t = t.view(-1, 1)
            if self.use_cond and cond is not None:
                x_in = torch.cat([x_t, cond, t], dim=1)
            else:
                x_in = torch.cat([x_t, t], dim=1)
            return self.net(x_in)

    class MalEncoder(nn.Module):
        def __init__(self, in_dim, emb_dim=LATENT_T):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.SiLU(),
                nn.Linear(256, emb_dim)
            )
        def forward(self, x): return self.net(x)

    from RQ3_DR_Synth_CondDiff_STP import DDPM, light_stp_loss
    ddpm = DDPM(T=TIMESTEPS)
    eps_model = EpsModelAbl(dim, LATENT_T, use_cond=USE_COND).to(device)
    enc = MalEncoder(dim, LATENT_T).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(eps_model.parameters()), lr=LR)
    db = torch.utils.data.DataLoader(torch.tensor(X_ben, dtype=torch.float32), batch_size=512, shuffle=True, drop_last=True)
    dm = torch.utils.data.DataLoader(torch.tensor(X_mal, dtype=torch.float32), batch_size=512, shuffle=True, drop_last=True)

    # 训练
    for ep in range(EPOCHS):
        total = 0
        for xb, xm in zip(db, cycle(dm)):
            xb, xm = xb.to(device), xm.to(device)
            B = min(xb.size(0), xm.size(0))
            xb, xm = xb[:B], xm[:B]
            cond = enc(xm)
            t = torch.randint(0, ddpm.T, (B,), device=device)
            xt, eps = ddpm.q_sample(xb, t)
            t_norm = (t.float()/ddpm.T).view(-1,1)
            eps_pred = eps_model(xt, t_norm, cond if USE_COND else None)
            loss = torch.mean((eps - eps_pred)**2)

            if USE_STP_LOSS and LAMBDA_HSIC>0:
                a_bar_t = DDPM._extract(ddpm.a_bar, t, xb.shape)
                x0_pred = (xt - (1.0 - a_bar_t).sqrt() * eps_pred) / (a_bar_t.sqrt() + 1e-8)

                # 灵活选择 STP 对
                pairs_all = {
                    "ALL": [(idxT, idxS), (idxS, idxP), (idxT, idxP)],
                    "TS":  [(idxT, idxS)],
                    "SP":  [(idxS, idxP)],
                    "TP":  [(idxT, idxP)],
                }
                loss_stp = 0; valid = 0
                for A,B in pairs_all[STP_PAIR_MODE]:
                    if len(A)==0 or len(B)==0: continue
                    valid += 1
                    loss_stp += torch.abs(
                        torch.mean(torch.abs((x0_pred[:,A]-x0_pred[:,A].mean(0)) *
                                             (x0_pred[:,B]-x0_pred[:,B].mean(0)))) -
                        torch.mean(torch.abs((xb[:,A]-xb[:,A].mean(0)) *
                                             (xb[:,B]-xb[:,B].mean(0))))
                    )
                if valid>0: loss += LAMBDA_HSIC * loss_stp / valid

            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"  [Ep{ep:02d}] Loss={total/len(db):.4f}")

    # 采样
    from RQ3_DR_Synth_CondDiff_STP import sample_adv_from_mal
    adv = sample_adv_from_mal(eps_model, enc, ddpm, X_mal)

    # 评估
    metrics = calculate_metrics(X_ben, adv, cols, idxT, idxS, idxP)
    return metrics


# -------------------------------
# 主函数：自动运行多组实验
# -------------------------------
def main():
    cfg_path = "config.yaml"
    variants = [
        ("Full (Ours)", {"LAMBDA_HSIC":0.05,"USE_STP_LOSS":True,"STP_PAIR_MODE":"ALL","USE_COND":True,"TIMESTEPS":400}),
        ("w/o STP", {"LAMBDA_HSIC":0.0,"USE_STP_LOSS":False}),
        ("TS-only", {"LAMBDA_HSIC":0.05,"STP_PAIR_MODE":"TS"}),
        ("SP-only", {"LAMBDA_HSIC":0.05,"STP_PAIR_MODE":"SP"}),
        ("TP-only", {"LAMBDA_HSIC":0.05,"STP_PAIR_MODE":"TP"}),
        ("λ=0.01", {"LAMBDA_HSIC":0.01}),
        ("λ=0.20", {"LAMBDA_HSIC":0.20}),
        ("No Cond", {"USE_COND":False}),
        ("T=200", {"TIMESTEPS":200}),
        ("T=100", {"TIMESTEPS":100}),
    ]
    results = {}
    for name, args in variants:
        m = run_experiment(cfg_path, name, **args)
        results[name] = m

    # 输出结果表格
    print("\n=== Ablation Summary ===")
    print("{:<15} {:>8} {:>11} {:>10} {:>10} {:>11} {:>11} {:>11}".format(
        "Variant","L2","Wass.","Pearson","CorrΔ","Δ_TS","Δ_SP","Δ_TP"))
    for k,v in results.items():
        print("{:<15} {:>8.3f} {:>11.3f} {:>10.3f} {:>10.3f} {:>11.3f} {:>11.3f} {:>11.3f}".format(
            k,v["L2"],v["Wasserstein"],v["Pearson"],v["CorrΔ"],
            v["STPΔ(T↔S)"],v["STPΔ(S↔P)"],v["STPΔ(T↔P)"]))

    # 保存为 LaTeX 文件
    os.makedirs("results", exist_ok=True)
    latex_path = "results/ablation_table.tex"
    with open(latex_path,"w",encoding="utf-8") as f:
        f.write("\\begin{table*}[!t]\n\\centering\n")
        f.write("\\caption{Ablation study on STP regularization, conditioning, and diffusion schedule.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\renewcommand{\\arraystretch}{1.2}\n")
        f.write("\\setlength{\\tabcolsep}{5pt}\n")
        f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write("Variant & L2$\\downarrow$ & Wass.$\\downarrow$ & Pearson$\\uparrow$ & Corr$\\Delta$$\\downarrow$ & "
                "$\\Delta_{T\\!S}\\downarrow$ & $\\Delta_{S\\!P}\\downarrow$ & $\\Delta_{T\\!P}\\downarrow$\\\\\n\\midrule\n")
        for k,v in results.items():
            f.write(f"{k} & {v['L2']:.3f} & {v['Wasserstein']:.3f} & {v['Pearson']:.3f} & {v['CorrΔ']:.3f} & "
                    f"{v['STPΔ(T↔S)']:.3f} & {v['STPΔ(S↔P)']:.3f} & {v['STPΔ(T↔P)']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"\n✅ 已将 LaTeX 表格输出至 {latex_path}")

if __name__ == "__main__":
    main()
