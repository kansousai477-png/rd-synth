import os, time, yaml, random, warnings
import numpy as np
import pandas as pd
import torch
from RQ3_DR_Synth_CondDiff_STP import (
    load_and_scale,
    split_feature_blocks,
    calculate_metrics,
    plot_domain_projection,
    plot_corr_heatmaps,
    train_cond_diffusion,
    sample_adv_from_mal,
    device,
)

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULT_DIR = "../results/ablation"
PLOT_DIR   = os.path.join(RESULT_DIR, "plots")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================================================
# Helper: unified experiment runner
# ==============================================================
def run_ablation_variant(name, X_ben, X_mal, cols, idxT, idxS, idxP,
                         lambda_reg=0.05, steps=400, cond=True, variant="full"):
    """Train and evaluate one DR-Synth variant with given configuration."""
    print(f"\n[Run] Variant={name} | λ={lambda_reg} | T={steps} | cond={cond} | mode={variant}")
    import RQ3_DR_Synth_CondDiff_STP as drs  # reload dynamically to modify globals
    drs.LAMBDA_HSIC = lambda_reg
    drs.TIMESTEPS = steps
    drs.USE_STP_LOSS = (lambda_reg > 0)
    drs.device = device

    # Disable conditional encoder if cond=False
    if not cond:
        def train_cond_diffusion_uncond(X_b, X_m, idxT, idxS, idxP, steps=400):
            """
            Unconditional diffusion training (for ablation: No Cond)
            Drops malicious conditioning; trains diffusion purely on benign reconstruction.
            """
            from RQ3_DR_Synth_CondDiff_STP import DDPM, NPDataset
            dim = X_b.shape[1]

            # Define a reduced epsilon model that only takes (x_t, t)
            class EpsModelUncond(torch.nn.Module):
                def __init__(self, dim_x, hidden=256):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(dim_x + 1, hidden),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden, hidden),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden, dim_x)
                    )

                def forward(self, x_t, t):
                    t = t.view(-1, 1).to(x_t.dtype)
                    return self.net(torch.cat([x_t, t], dim=1))

            eps_model = EpsModelUncond(dim).to(device)
            ddpm = DDPM(T=steps)
            opt = torch.optim.Adam(eps_model.parameters(), lr=1e-3)
            from torch.utils.data import DataLoader
            db = DataLoader(NPDataset(X_b), batch_size=512, shuffle=True, drop_last=True)

            for ep in range(10):
                total = 0.0
                for xb in db:
                    xb = xb.to(device)
                    B = xb.size(0)
                    t = torch.randint(0, ddpm.T, (B,), device=device)
                    xt, eps = ddpm.q_sample(xb, t)
                    t_norm = (t.float() / ddpm.T).view(-1, 1)
                    eps_pred = eps_model(xt, t_norm)
                    loss = torch.mean((eps - eps_pred) ** 2)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += loss.item()
                print(f"[Uncond Epoch {ep:03d}] Loss={total / len(db):.6f}")

            enc = None  # no conditioning encoder
            return enc, eps_model, ddpm

        enc, eps_model, ddpm = train_cond_diffusion_uncond(X_ben, X_mal, idxT, idxS, idxP)
    else:
        enc, eps_model, ddpm = train_cond_diffusion(X_ben, X_mal, idxT, idxS, idxP)

    adv = sample_adv_from_mal(eps_model, enc, ddpm, X_mal)
    metrics = calculate_metrics(X_ben, adv, cols, idxT, idxS, idxP)

    np.save(os.path.join(RESULT_DIR, f"{name}_adv.npy"), adv)
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, f"{name}_metrics.csv"), index=False)
    plot_dir = os.path.join(PLOT_DIR, name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_domain_projection(X_mal, adv, X_ben, save_dir=plot_dir)
    plot_corr_heatmaps(X_ben, adv, cols, save_dir=plot_dir)
    print(f"✅ {name} done.\n")
    return metrics


# ==============================================================
# Main routine
# ==============================================================
def main():
    with open('config.yaml','r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ben_path = cfg['labeled_benign_csv']
    mal_path = cfg['labeled_malicious_csv']

    X_ben, X_mal, cols = load_and_scale(ben_path, mal_path)
    idxT, idxS, idxP = split_feature_blocks(cols)
    print(f"[Info] D={len(cols)}, T={len(idxT)}, S={len(idxS)}, P={len(idxP)}")

    results = {}

    # === Table 1: Objective variants ===
    results["Full (Ours)"] = run_ablation_variant("full", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.05)
    results["w/o reg"]     = run_ablation_variant("no_reg", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.0)
    results["T-S only"]    = run_ablation_variant("ts_only", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.05, variant="ts")
    results["S-P only"]    = run_ablation_variant("sp_only", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.05, variant="sp")
    results["T-P only"]    = run_ablation_variant("tp_only", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.05, variant="tp")

    # === Table 2: Hyperparameter variants ===
    results["Weaker λ=0.01"]   = run_ablation_variant("lambda_001", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.01)
    results["Stronger λ=0.20"] = run_ablation_variant("lambda_020", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.20)
    results["No Cond"]         = run_ablation_variant("no_cond", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.05, cond=False)
    results["T=200"]           = run_ablation_variant("steps_200", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.05, steps=200)
    results["T=100"]           = run_ablation_variant("steps_100", X_ben, X_mal, cols, idxT, idxS, idxP, lambda_reg=0.05, steps=100)

    # === Aggregate all results ===
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Variant"
    df.to_csv(os.path.join(RESULT_DIR, "ablation_metrics_all.csv"))
    print(f"\n✅ All ablation results saved to {RESULT_DIR}/ablation_metrics_all.csv")

if __name__ == "__main__":
    main()
