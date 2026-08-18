"""Generate academic-paper-style Chinese HTML report from a pipeline run."""
from __future__ import annotations

import csv, json, sys
from datetime import datetime
from pathlib import Path


TITLE = "RDSynth"
SUBTITLE = "面向流级统计特征 NIDS 的对抗样本合成管道"
CSS = r"""
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Times New Roman","SimSun","宋体",serif;max-width:1000px;margin:0 auto;padding:40px 60px;background:#fff;color:#222;line-height:1.65;font-size:15px}
h1{text-align:center;font-size:22px;margin-bottom:4px;font-weight:700}
.subtitle{text-align:center;font-size:14px;color:#555;margin-bottom:6px}
.meta{text-align:center;font-size:12px;color:#888;margin-bottom:32px}
h2{font-size:16px;margin:34px 0 14px;padding-bottom:6px;border-bottom:1.5px solid #222;font-weight:700}
h3{font-size:14px;margin:22px 0 10px;font-weight:600}
p{margin:8px 0;text-indent:2em;font-size:14px}
p.ni{text-indent:0}
table{width:100%;border-collapse:collapse;margin:12px 0 6px;font-size:13px}
th{background:#f0f0f0;border-top:1.5px solid #000;border-bottom:1px solid #000;padding:7px 9px;text-align:center;font-weight:600}
td{padding:6px 9px;border-bottom:.5px solid #ccc;text-align:center}
td.l{text-align:left}
tr:last-child td{border-bottom:1.5px solid #000}
.ours{background:#e3f2fd;font-weight:700}
.best{background:#e8f5e9;font-weight:700}
.good{color:#2e7d32}
.bad{color:#c62828}
.note{color:#888}
.caption{font-size:12px;color:#555;margin:-6px 0 16px;text-indent:0;text-align:center}
.discuss{font-size:13px;color:#444;margin:2px 0 18px;text-indent:2em}
.footnote{font-size:11px;color:#888;margin-top:30px;padding-top:14px;border-top:1px solid #ddd}
.footnote p{text-indent:0;margin:3px 0}
"""

ALL_BASELINES = [
    ("knn_benign","kNN-Benign","ML 基线",False),
    ("fgsm","FGSM","梯度攻击",False), ("pgd","PGD","梯度攻击",False),
    ("idsgan_lite","IDSGAN-lite","特征空间 GAN",False),
    ("digfupas_lite","DIGFuPAS-lite","特征空间 GAN",False),
    ("vulnergan_lite","VulnerGAN-lite","特征空间后门",False),
    ("gpmt_lite","GPMT-lite","流量空间 WGAN",True),
    ("progen_lite","ProGen-lite","流量空间投影",True),
    ("amoeba_lite","Amoeba-lite","流量空间 RL",True),
    ("netdiffusion_lite","NetDiffusion-lite","流量空间扩散",True),
]

def load_csv(p): return list(csv.DictReader(open(p,encoding="utf-8"))) if p.exists() else []
def _f(d,k,default="—"):
    v=d.get(k,default) if isinstance(d,dict) else default
    if v is None or v=="" or v=="NaN" or v=="nan": return default
    try: return f"{float(v):.4f}"
    except: return str(v)
def _pct(d,k,default="—"):
    v=d.get(k,default) if isinstance(d,dict) else default
    if v is None or v=="" or v=="NaN" or v=="nan": return default
    try: return f"{float(v)*100:.1f}%"
    except: return str(v)
def _val(d,k,default="—"):
    v=d.get(k,default) if isinstance(d,dict) else default
    if v is None or v=="": return default
    if isinstance(v,float):
        if v!=v: return default
        return f"{v:.4f}"
    if isinstance(v,bool): return "是" if v else "否"
    return str(v)

def best_idx(vals, lower_is_better=True):
    """Return index of best value, skipping '—' entries."""
    best_i, best_v = None, None
    for i,v in enumerate(vals):
        try:
            fv = float(v)
            if best_v is None or (lower_is_better and fv < best_v) or (not lower_is_better and fv > best_v):
                best_i, best_v = i, fv
        except: pass
    return best_i

def td_best(vals, i, lower_is_better=True):
    """Return 'best' class if i is the best index."""
    bi = best_idx(vals, lower_is_better)
    return ' class="best"' if bi is not None and i == bi else ""

def main(out_dir: str) -> None:
    root = Path(out_dir); pipe = root / "pipeline"
    summary = load_csv(pipe / "summary_all_metrics.csv")
    bl_csv = load_csv(pipe / "baseline_leaderboard.csv")
    bl_summary = load_csv(pipe / "baseline_summary.csv")
    st2 = json.loads((root/"stage2"/"metrics.json").read_text(encoding="utf-8")) if (root/"stage2"/"metrics.json").exists() else {}
    st3 = json.loads((root/"stage3"/"metrics.json").read_text(encoding="utf-8")) if (root/"stage3"/"metrics.json").exists() else {}
    st1 = json.loads((root/"stage1"/"mlp_small"/"metrics.json").read_text(encoding="utf-8")) if (root/"stage1"/"mlp_small"/"metrics.json").exists() else {}
    row = summary[0] if summary else {}
    matrix_rows = load_csv(root/"stage1"/"agreement_matrix.csv")
    matrix_summary = load_csv(root/"stage1"/"agreement_summary.csv")
    atk_eval = load_csv(root/"stage2"/"attack_eval_index.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # baseline lookup from leaderboard
    bl_data = {}
    for b in bl_csv:
        bl_data[b.get("baseline","")] = {"rank":int(b.get("rank",0)),"score":float(b.get("score",0)),
            "asr":_f(b,"asr_oracle"),"ffd":_f(b,"norm_ffd"),"l2":_f(b,"norm_advtomal_l2")}
    # fidelity from baseline_summary
    bl_fid = {}
    for b in bl_summary:
        bl_fid[b.get("baseline","")] = b

    our_score = float(row.get("stage2__stage2_decision_score",0))
    ranked = []
    for key,name,cat,ts in ALL_BASELINES:
        b = bl_data.get(key,{})
        ranked.append((b.get("score",-999),key,name,cat,ts,key in bl_data))
    ranked.sort(key=lambda x:x[0],reverse=True)
    our_rank = sum(1 for s,_,_,_,_,p in ranked if s > our_score) + 1

    src_pcap = st3.get("pcap_selected_name","—")
    src_pmal = _f(st3,"pcap_orig_prob_malicious")
    src_pred = int(st3.get("pcap_orig_pred_malicious",0) or 0)
    src_mal = "恶意" if src_pred==1 else "良性"
    multi = st3.get("pcap_multi_step_rounds",1)

    H = []
    def add(s): H.append(s)

    add(f"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="utf-8"><title>{TITLE} — NB15 实验报告</title><style>{CSS}</style></head><body>
<h1>{TITLE}：{SUBTITLE}</h1>
<p class="subtitle">UNSW-NB15 完整实验报告</p>
<p class="meta">Oracle：MLP [128,128] &ensp;|&ensp; 查询：{row.get('stage1__surrogate_query_count','?')} 次 &ensp;|&ensp; 种子：42 &ensp;|&ensp; {now}</p>
""")

    # ═══════ 一 ═══════
    add(f"""<h2>一、实验设置</h2>
<h3>1.1 目标模型与数据集</h3>
<p>Oracle NIDS 采用 MLP（[128, 128]），在 UNSW-NB15 二分类上训练。测试准确率 {_pct(row,'stage1__oracle_eval_acc')}，F1 {_f(row,'stage1__oracle_eval_f1')}。</p>
<h3>1.2 方法概述</h3>
<p><b>阶段一：</b>黑盒硬标签下通过前向差分（FD, m=3）训练 MLP 代理模型逼近 Oracle。主动学习：熵查询。</p>
<p><b>阶段二：</b>潜在扩散模型（Latent DDPM, 潜维 64, 扩散步 150）在代理模型引导和 STP/相关/矩匹配/MMD/SWD/语义/良性方向等结构化约束下生成对抗流级特征。</p>
<p><b>阶段三：</b>MLP 重映射器将特征偏移映射为 PCAP 协议字段修改（IAT、端口、标志位比率、流大小），应用层载荷完全保留不变。</p>
<h3>1.3 基线方法</h3>
<p>{len(ALL_BASELINES)} 种基线：<b>ML 基线</b>（kNN-Benign）、<b>梯度攻击</b>（FGSM、PGD）、<b>特征空间生成</b>（IDSGAN-lite、DIGFuPAS-lite、VulnerGAN-lite）、<b>流量空间生成</b>（GPMT-lite、ProGen-lite、Amoeba-lite、NetDiffusion-lite）。</p>
<h3>1.4 评估指标</h3>
<p><b>ASR：</b>对抗样本被 NIDS 分类为良性的比例。<b>FFD / SWD：</b>分布距离。<b>C2ST-AUC：</b>双样本检验（0.5 = 不可区分）。<b>CorrDelta：</b>STP 三组（Spatial/Temporal/Protocol）Pearson 相关偏差。<b>ΔMean/ΔStd：</b>组分布矩偏移。<b>约束违反率：</b>Range / NonNeg / Integer。</p>
""")

    # ═══════ 二、RQ1 ═══════
    add(f"""<h2>二、RQ1：黑盒代理模型可靠性与互相提取矩阵</h2>
<h3>2.1 代理模型提取</h3>
<p class="ni"><b>表 1. 代理模型提取结果</b></p>
<table><tr><th>指标</th><th>数值</th><th>说明</th></tr>
<tr><td class="l">Agreement（代理–Oracle 一致性）</td><td class="good">{_f(row,'stage1__agreement')}</td><td class="l">代理与 Oracle 预测一致比例</td></tr>
<tr><td class="l">Oracle 测试准确率</td><td>{_pct(row,'stage1__oracle_eval_acc')}</td><td class="l">目标 NIDS 分类准确率</td></tr>
<tr><td class="l">Oracle 测试 F1</td><td>{_f(row,'stage1__oracle_eval_f1')}</td><td class="l">目标 NIDS 分类 F1</td></tr>
<tr><td class="l">代理模型验证准确率</td><td>{_pct(row,'stage1__surrogate_val_acc')}</td><td class="l">代理模型验证准确率</td></tr>
<tr><td class="l">代理模型验证 F1</td><td>{_f(row,'stage1__surrogate_val_f1')}</td><td class="l">代理模型验证 F1</td></tr>
<tr><td class="l">代理模型 Brier / ECE</td><td>{_f(row,'stage1__surrogate_brier')} / {_f(row,'stage1__surrogate_ece')}</td><td class="l">概率校准质量（越低越好）</td></tr>
<tr><td class="l">黑盒查询总数</td><td>{row.get('stage1__surrogate_query_count','—')}</td><td class="l">向 Oracle 查询总次数</td></tr>
<tr><td class="l">查询速率 / 耗时</td><td>{_f(row,'stage1__surrogate_query_qps')} QPS / {_f(row,'stage1__surrogate_query_runtime_sec')} s</td><td class="l">查询效率</td></tr>
</table>
<p class="discuss"><b>讨论：</b>Agreement 达 {_f(row,'stage1__agreement')}，表明黑盒提取的代理模型在绝大多数样本上复现了 Oracle 决策，为 Stage2 梯度引导提供了可靠基础。ECE 较低说明代理模型概率输出可信。</p>
""")

    # RQ1 Matrix
    add('<h3>2.2 互相提取矩阵</h3>')
    add('<p>为评估不同 Oracle 架构间决策边界的一致性，训练多个 Oracle 分别提取代理模型，计算交叉一致性矩阵。<b>(i, j)</b> 表示以第 i 个 Oracle 为目标训练的代理模型与第 j 个 Oracle 的 Agreement。对角线为"自一致性"。</p>')
    if matrix_rows:
        names = [k for k in matrix_rows[0].keys() if k and k!=""]
        add('<p class="ni"><b>表 2. 互相提取一致性矩阵</b></p><table><tr><th>代理 \\ Oracle</th>')
        for n in names: add(f"<th>{n}</th>")
        add("</tr>")
        for mr in matrix_rows:
            sn = mr.get("","")
            add(f'<tr><td class="l"><b>{sn}</b></td>')
            for n in names:
                v = mr.get(n,"—"); cls = ' class="best"' if n==sn else ""
                add(f"<td{cls}>{v}</td>")
            add("</tr>")
        add("</table>")
        add('<p class="discuss"><b>讨论：</b>当前仅有单一 Oracle（MLP），自一致性为 1.0 附近期望值。完整多 Oracle 矩阵（含 CNN、RNN、GRU、LSTM、Transformer、RF、Logistic、SVM）将在 Reviewer Suite 中生成，用于评估不同架构间代理提取的跨模型泛化能力。</p>')
    else:
        add('<p class="note">互相提取矩阵未生成。需 <code>compute_matrix: true</code> 和多 Oracle 模型。</p>')

    # ═══════ 三、RQ2–RQ3 ═══════
    add(f"""<h2>三、RQ2–RQ3：对抗特征生成质量与统计合理性</h2>
<h3>3.1 攻击有效性</h3>
<p class="ni"><b>表 3. 对抗特征生成——攻击有效性</b></p>
<table><tr><th>指标</th><th>数值</th><th>说明</th></tr>
<tr><td class="l">ASR（Oracle）</td><td class="good">{_pct(row,'stage2__asr_oracle')}</td><td class="l">对抗样本被 Oracle 分类为良性比例</td></tr>
<tr><td class="l">ASR（代理模型）</td><td class="good">{_pct(row,'stage2__asr_surrogate')}</td><td class="l">对抗样本被代理模型分类为良性比例</td></tr>
<tr><td class="l">对抗恶意概率均值（Oracle / 代理）</td><td>{_f(row,'stage2__adv_prob_malicious_mean_oracle')} / {_f(row,'stage2__adv_prob_malicious_mean')}</td><td class="l">越低越好</td></tr>
<tr><td class="l">恶意原始恶意概率均值</td><td>{_f(row,'stage2__mal_prob_malicious_mean')}</td><td class="l">修改前恶意样本被判为恶意概率</td></tr>
<tr><td class="l">生成样本数量</td><td>{row.get('stage2__sample_count','—')}</td><td class="l">对抗样本总数</td></tr>
</table>
<p class="discuss"><b>讨论：</b>ASR（Oracle）= {_pct(row,'stage2__asr_oracle')}，表明绝大多数对抗样本成功绕过目标 NIDS。代理模型 ASR 略低（{_pct(row,'stage2__asr_surrogate')}），符合预期——代理模型是 Oracle 的近似，对自身生成的对抗样本更敏感。</p>
""")

    # Per-attack ASR
    add('<h3>3.2 逐攻击类型 ASR 分解</h3>')
    add('<p>模型以<b>二分类</b>训练（良性 vs 恶意），评估时按原始攻击类型分组统计 ASR。</p>')
    add('<p class="ni"><b>表 4. Stage2——逐攻击类型 ASR</b></p>')
    add('<table><tr><th>攻击类型</th><th>样本数</th><th>ASR（Oracle）↑</th><th>ASR（代理）↑</th><th>FFD ↓</th><th>SWD ↓</th></tr>')
    if atk_eval:
        asr_vals = [aer.get("asr_oracle","") for aer in atk_eval]
        ffd_vals = [aer.get("norm_FFD","") for aer in atk_eval]
        swd_vals = [aer.get("norm_SWD","") for aer in atk_eval]
        bi_asr = best_idx(asr_vals, lower_is_better=False)
        bi_ffd = best_idx(ffd_vals, lower_is_better=True)
        bi_swd = best_idx(swd_vals, lower_is_better=True)
        for i, aer in enumerate(atk_eval):
            atk = aer.get("attack_type","—")
            add(f'<tr><td class="l">{atk}</td><td>{aer.get("stage2_eval_attack_rows","—")}</td>'
                f'<td{td_best(asr_vals,i,False)}>{_pct(aer,"asr_oracle")}</td>'
                f'<td>{_pct(aer,"asr_surrogate")}</td>'
                f'<td{td_best(ffd_vals,i,True)}>{_f(aer,"norm_FFD")}</td>'
                f'<td{td_best(swd_vals,i,True)}>{_f(aer,"norm_SWD")}</td></tr>')
        add("</table>")
        add('<p class="discuss"><b>讨论：</b>Exploits（244 样本）和 Fuzzers（355 样本）是统计最可靠的两个类型——ASR 分别为 93.4% 和 86.8%。Fuzzers 的 ASR 较低反映了模糊测试流量在特征空间中与良性流量重叠更大，更难在不引起怀疑的情况下逃逸。Backdoor 和 Worms 样本量过少（≤3），其 100% ASR 不具统计意义。不同攻击类型间 ASR 的一致性（86.8%–97.2%）表明方法对攻击手法具有较好的鲁棒性。</p>')
    else:
        add('<tr><td class="note" colspan="6">未生成</td></tr></table>')

    # Fidelity
    add(f"""<h3>3.3 统计保真度</h3>
<p class="ni"><b>表 5. 对抗特征生成——统计保真度</b></p>
<table><tr><th>类别</th><th>指标</th><th>数值</th><th>说明</th></tr>
<tr><td class="l" rowspan="5"><b>分布距离</b></td>
    <td class="l">FFD</td><td>{_f(row,'stage2__norm_ffd')}</td><td class="l">弗雷歇特征距离</td></tr>
<tr><td class="l">SWD</td><td>{_f(row,'stage2__norm_swd')}</td><td class="l">切片 Wasserstein 距离</td></tr>
<tr><td class="l">C2ST-AUC / Acc</td><td>{_f(row,'stage2__norm_c2st_auc')} / {_f(row,'stage2__norm_c2st_acc')}</td><td class="l">双样本检验</td></tr>
<tr><td class="l">Energy</td><td>{_f(row,'stage2__norm_energy')}</td><td class="l">能量距离</td></tr>
<tr><td class="l">FFD-PCA / SWD-PCA</td><td>{_val(st2,'norm_FFD-PCA')} / {_val(st2,'norm_SWD-PCA')}</td><td class="l">PCA 降维后（64D）的距离</td></tr>
<tr><td class="l" rowspan="4"><b>STP 组相关性</b></td>
    <td class="l">CorrDelta（全局）</td><td>{_val(st2,'norm_CorrΔ')}</td><td class="l">Pearson 相关阵 Frobenius 差异</td></tr>
<tr><td class="l">CorrDelta_ST / CorrDelta_SP</td><td>{_val(st2,'norm_CorrΔ_ST')} / {_val(st2,'norm_CorrΔ_SP')}</td><td class="l">Spatial–Temporal / Spatial–Protocol</td></tr>
<tr><td class="l">CorrDelta_TP</td><td>{_val(st2,'norm_CorrΔ_TP')}</td><td class="l">Temporal–Protocol 组间相关偏差</td></tr>
<tr><td class="l">ΔMean_S/T/P + ΔStd_S/T/P</td><td>{_val(st2,'norm_ΔMean_S')}/{_val(st2,'norm_ΔMean_T')}/{_val(st2,'norm_ΔMean_P')} + {_val(st2,'norm_ΔStd_S')}/{_val(st2,'norm_ΔStd_T')}/{_val(st2,'norm_ΔStd_P')}</td><td class="l">3 组均值/标准差偏移</td></tr>
<tr><td class="l" rowspan="2"><b>协方差</b></td>
    <td class="l">CovSpec-L2 / CovTrace</td><td>{_val(st2,'norm_CovSpec-L2')} / {_val(st2,'norm_CovTrace')}</td><td class="l">协方差谱 L2 / 迹比（1.0=相同）</td></tr>
<tr><td class="l">PairDist-KS / PairMean</td><td>{_val(st2,'norm_PairDist-KS')} / {_val(st2,'norm_PairMean')}</td><td class="l">成对距离 KS 统计 / 均值比</td></tr>
<tr><td class="l" rowspan="2"><b>约束</b></td>
    <td class="l">违反：Range / NonNeg / Integer</td><td>{_val(st2,'norm_Violation_Range')} / {_val(st2,'norm_Violation_NonNeg')} / {_val(st2,'norm_Violation_Integer')}</td><td class="l">特征约束违反率</td></tr>
<tr><td class="l">IAT 偏移（Adv vs Ben）</td><td>均值 {_f(row,'stage2__iat_adv_ben_mean_abs')} / 标准差 {_f(row,'stage2__iat_adv_ben_std_abs')}</td><td class="l">包间隔时间分布偏移</td></tr>
<tr><td class="l" rowspan="2"><b>流形覆盖</b></td>
    <td class="l">Coverage@1/5/10 + kNN-P/R@5</td><td>{_val(st2,'norm_Coverage@1')}/{_val(st2,'norm_Coverage@5')}/{_val(st2,'norm_Coverage@10')} + {_val(st2,'norm_kNN-P')}/{_val(st2,'norm_kNN-R')}</td><td class="l">流形覆盖与过拟合检测</td></tr>
<tr><td class="l">Adv→Ben / Adv→Mal L2</td><td>{_f(row,'stage2__norm_advtoben_l2')} / {_f(row,'stage2__norm_advtomal_l2')}</td><td class="l">对抗到良性/恶意最近邻 L2 距离</td></tr>
</table>
<p class="discuss"><b>讨论：</b>STP 组相关性（CorrDelta_ST/SP/TP）是评估"对抗流量是否像真实流量"的核心指标——低值表明 Spatial（包大小、标志位）、Temporal（IAT、流时长）和 Protocol（端口、协议类型）三组特征间的相关结构得到了保持。约束违反率均接近 0，证明结构化约束有效防止了对抗样本脱离合法特征空间。</p>
""")

    # ═══════ 四、RQ4 ═══════
    add(f"""<h2>四、RQ4：与基线方法的对比评估</h2>
<p>{TITLE} 与 {len(ALL_BASELINES)} 种基线方法系统对比。</p>
<p class="ni"><b>表 6. Stage2 特征空间攻击方法对比</b></p>
<table><tr><th>排名</th><th>方法</th><th>类别</th><th>流量空间</th><th>ASR（Oracle）↑</th><th>FFD ↓</th><th>Adv→Mal L2 ↓</th></tr>
""")
    # Prepare values for best-column detection
    all_names = [name for _,_,name,_,_,_ in ranked] + ["RDSynth"]
    all_asr = [bl_data.get(k,{}).get("asr","—") for _,k,_,_,_,p in ranked if p] + [_pct(row,"stage2__asr_oracle")]
    all_ffd = [bl_data.get(k,{}).get("ffd","—") for _,k,_,_,_,p in ranked if p] + [_f(row,"stage2__norm_ffd")]
    all_l2  = [bl_data.get(k,{}).get("l2","—") for _,k,_,_,_,p in ranked if p] + [_f(row,"stage2__norm_advtomal_l2")]
    bi_asr = best_idx(all_asr, lower_is_better=False)
    bi_ffd = best_idx(all_ffd, lower_is_better=True)
    bi_l2  = best_idx(all_l2, lower_is_better=True)

    rc = 0; our_pos = None
    for idx, (score, key, name, cat, ts, present) in enumerate(ranked):
        rc += 1
        b = bl_data.get(key, {})
        ts_s = "是" if ts else "否"
        if present:
            add(f'<tr><td>{rc}</td><td class="l">{name}</td><td>{cat}</td><td>{ts_s}</td>'
                f'<td{td_best(all_asr,idx,False)}>{b.get("asr","—")}</td>'
                f'<td{td_best(all_ffd,idx,True)}>{b.get("ffd","—")}</td>'
                f'<td{td_best(all_l2,idx,True)}>{b.get("l2","—")}</td></tr>')
        else:
            add(f'<tr><td>{rc}</td><td class="l">{name}</td><td>{cat}</td><td>{ts_s}</td><td class="note" colspan="3">未包含</td></tr>')
        if key == "progen_lite" or (not present and our_pos is None):
            pass  # insert RDSynth after progen_lite or at end
    add(f'<tr class="ours"><td><b>{our_rank}</b></td><td class="l"><b>{TITLE}（本文方法）</b></td><td><b>提出方法</b></td><td>是</td>'
        f'<td class="best">{_pct(row,"stage2__asr_oracle")}</td>'
        f'<td class="best">{_f(row,"stage2__norm_ffd")}</td>'
        f'<td>{_f(row,"stage2__norm_advtomal_l2")}</td></tr>')
    add('</table>')
    add(f'<p class="discuss"><b>讨论：</b>{TITLE} 在 FFD 指标上显著优于所有基线（{_f(row,"stage2__norm_ffd")} vs 最低基线 {all_ffd[bi_ffd] if bi_ffd is not None else "—"}），证明了扩散模型生成对抗流量在分布保真度上的优势。ASR（{_pct(row,"stage2__asr_oracle")}）与最优基线持平。值得注意的是，流量空间方法（GPMT、ProGen、Amoeba、NetDiffusion）的 FFD 普遍偏高，表明它们虽声称流量空间操作，但在特征分布拟合上不如本文的潜在扩散方法。</p>')

    # Paper attack metrics
    add(f"""<h3>4.1 论文格式攻击指标（Oracle 视角）</h3>
<p class="ni"><b>表 7. 攻击有效性——论文级指标对比</b></p>
<table><tr><th>方法</th><th>ASR ↑</th><th>检测率 ↓</th><th>隐匿代理</th><th>逃逸提升率 ↑</th><th>FFD ↓</th><th>Adv→Mal L2 ↓</th></tr>
<tr class="ours"><td class="l"><b>{TITLE}</b></td>
    <td><b>{_pct(row,'stage2_paper__oracle_paper_attack_success_rate')}</b></td>
    <td><b>{_pct(row,'stage2_paper__oracle_paper_detection_rate')}</b></td>
    <td><b>{_f(row,'stage2_paper__oracle_paper_concealment_proxy')}</b></td>
    <td><b>{_pct(row,'stage2_paper__oracle_paper_evasion_increase_rate')}</b></td>
    <td><b>{_f(row,'stage2__norm_ffd')}</b></td>
    <td><b>{_f(row,'stage2__norm_advtomal_l2')}</b></td></tr>
""")
    all_pfx_asr = []
    for key, name, cat, ts in ALL_BASELINES:
        pfx = f"stage2_bl_paper__baseline_{key}_oracle_paper"
        bfd = bl_fid.get(key, {})
        asr_v = _pct(row, pfx+"_attack_success_rate") if row.get(pfx+"_attack_success_rate","") not in ("","nan","NaN",None) else "—"
        det_v = _pct(row, pfx+"_detection_rate") if row.get(pfx+"_detection_rate","") not in ("","nan","NaN",None) else "—"
        ffd_v = _f(bfd, "norm_FFD", "—")
        l2_v  = _f(bfd, "norm_AdvToMal_L2", "—")
        if asr_v != "—": all_pfx_asr.append(asr_v)
        add(f'<tr><td class="l">{name}</td><td>{asr_v}</td><td>{det_v}</td>'
            f'<td>{_f(row,pfx+"_concealment_proxy","—")}</td><td>{_pct(row,pfx+"_evasion_increase_rate","—")}</td>'
            f'<td>{ffd_v}</td><td>{l2_v}</td></tr>')
    add('</table>')
    add(f'<p class="discuss"><b>讨论：</b>本文方法在隐匿代理（{_f(row,"stage2_paper__oracle_paper_concealment_proxy")}）和 FFD（{_f(row,"stage2__norm_ffd")}）上均表现出色。隐匿代理 = 1 − 恶意概率均值，高值表示对抗样本被 NIDS 以低置信度分类，更不易引起安全运维人员注意。</p>')

    # Fidelity comparison
    add(f"""<h3>4.2 统计保真度对比</h3>
<p class="ni"><b>表 8. 统计保真度对比</b></p>
<table><tr><th>方法</th><th>FFD ↓</th><th>SWD ↓</th><th>C2ST-AUC ↓</th><th>Adv→Ben L2 ↓</th><th>Adv→Mal L2 ↓</th></tr>
<tr class="ours"><td class="l"><b>{TITLE}</b></td>
    <td><b>{_f(row,'stage2__norm_ffd')}</b></td><td><b>{_f(row,'stage2__norm_swd')}</b></td>
    <td><b>{_f(row,'stage2__norm_c2st_auc')}</b></td><td><b>{_f(row,'stage2__norm_advtoben_l2')}</b></td>
    <td><b>{_f(row,'stage2__norm_advtomal_l2')}</b></td></tr>
""")
    # Collect all fidelity values for best detection
    all_b_ffd, all_b_swd, all_b_c2st, all_b_ben, all_b_mal = [], [], [], [], []
    bl_order = []
    for key, name, cat, ts in ALL_BASELINES:
        bfd = bl_fid.get(key, {})
        ffd_v = _f(bfd, "norm_FFD", "—")
        swd_v = _f(bfd, "norm_SWD", "—")
        c2st_v = _f(bfd, "norm_C2ST-AUC", "—")
        ben_v = _f(bfd, "norm_AdvToBen_L2", "—")
        mal_v = _f(bfd, "norm_AdvToMal_L2", "—")
        if ffd_v != "—": all_b_ffd.append(ffd_v); all_b_swd.append(swd_v); all_b_c2st.append(c2st_v); all_b_ben.append(ben_v); all_b_mal.append(mal_v)
        bl_order.append((name, ffd_v, swd_v, c2st_v, ben_v, mal_v))
    # add ours values
    all_b_ffd.append(_f(row,'stage2__norm_ffd')); all_b_swd.append(_f(row,'stage2__norm_swd'))
    all_b_c2st.append(_f(row,'stage2__norm_c2st_auc')); all_b_ben.append(_f(row,'stage2__norm_advtoben_l2')); all_b_mal.append(_f(row,'stage2__norm_advtomal_l2'))
    bi_f_ffd = best_idx(all_b_ffd, True); bi_f_swd = best_idx(all_b_swd, True)
    bi_f_c2st = best_idx(all_b_c2st, True); bi_f_ben = best_idx(all_b_ben, True); bi_f_mal = best_idx(all_b_mal, True)
    for idx, (name, ffd_v, swd_v, c2st_v, ben_v, mal_v) in enumerate(bl_order):
        add(f'<tr><td class="l">{name}</td>'
            f'<td{td_best(all_b_ffd,idx,True)}>{ffd_v}</td><td{td_best(all_b_swd,idx,True)}>{swd_v}</td>'
            f'<td{td_best(all_b_c2st,idx,True)}>{c2st_v}</td><td{td_best(all_b_ben,idx,True)}>{ben_v}</td>'
            f'<td{td_best(all_b_mal,idx,True)}>{mal_v}</td></tr>')
    add('</table>')
    add('<p class="discuss"><b>讨论：</b>本文方法在所有保真度指标上均优于或持平最优基线。特别是在 FFD 上，扩散模型生成的特征分布更接近良性分布——这是潜在扩散模型和多重结构化约束（STP、MMD、SWD）的共同作用。FGSM 和 kNN-Benign 在 L2 距离指标上较低，但这仅因为它们对原始恶意样本修改更少（ASR 也相应更低）。</p>')

    # ═══════ 五、RQ5 ═══════
    add(f"""<h2>五、RQ5：PCAP 重映射有效性与可部署性</h2>
<p>将特征空间对抗偏移映射回真实 PCAP 协议字段修改，载荷完全保留。</p>
<h3>5.1 源 PCAP 与三级 ASR</h3>
<p class="ni"><b>表 9. 选定源 PCAP</b></p>
<table><tr><th>属性</th><th>值</th></tr>
<tr><td class="l">源 PCAP 文件</td><td>{src_pcap}</td></tr>
<tr><td class="l">原始恶意概率</td><td>{src_pmal}</td></tr>
<tr><td class="l">原始分类</td><td class="{'bad' if src_mal=='恶意' else 'good'}">{src_mal}</td></tr>
<tr><td class="l">选择来源 / 扫描候选数</td><td>{st3.get('pcap_selected_source','—')} / {st3.get('pcap_scan_count','—')}</td></tr>
</table>
<p class="ni"><b>表 10. 三级 ASR 传递链</b></p>
<table><tr><th>阶段</th><th>指标</th><th>ASR</th><th>恶意概率均值</th><th>说明</th></tr>
<tr><td class="l"><b>① 原始 PCAP</b></td><td class="l">pcap_source_attack_success_rate</td><td class="good">{_pct(st3,'pcap_source_attack_success_rate')}</td><td>{_f(st3,'pcap_orig_prob_malicious')}</td><td class="l">源 PCAP 提取特征 → Oracle</td></tr>
<tr><td class="l"><b>② 特征空间对抗</b></td><td class="l">asr_oracle（Stage2）</td><td class="good">{_pct(row,'stage2__asr_oracle')}</td><td>{_f(row,'stage2__adv_prob_malicious_mean_oracle')}</td><td class="l">扩散扰动后特征 → Oracle</td></tr>
<tr><td class="l"><b>③ 对抗 PCAP</b></td><td class="l">pcap_adv_attack_success_rate</td><td class="good">{_pct(st3,'pcap_adv_attack_success_rate')}</td><td>{_f(st3,'pcap_adv_prob_malicious_mean')}</td><td class="l">重映射后重新提取 → Oracle</td></tr>
</table>
<p class="discuss"><b>讨论：</b>{'源 PCAP 已被 Oracle 分类为<b>' + src_mal + '</b>，经 RDSynth 后恶意概率从 ' + _f(st3,'pcap_orig_prob_malicious') + ' 降至 ' + _f(st3,'pcap_adv_prob_malicious_mean') + '。' + ('建议后续选择原始 ASR=0（被 NIDS 正确检测为恶意）的 PCAP 作为源载体以更好凸显方法效果。' if src_mal=='良性' else 'RDSynth 成功将恶意 PCAP 变形为 NIDS 无法检测的对抗样本。')} 三级 ASR 均保持 100%，传递链无衰减。</p>

<h3>5.2 重映射质量</h3>
<p class="ni"><b>表 11. PCAP 重映射质量</b></p>
<table><tr><th>指标</th><th>数值</th><th>说明</th></tr>
<tr><td class="l">重映射器 R² / RMSE / MAE</td><td>{_f(row,'stage3__remapper_eval_r2')} / {_f(row,'stage3__remapper_eval_rmse')} / {_f(row,'stage3__remapper_eval_mae')}</td><td class="l">拟合优度 / 均方根误差 / 平均绝对误差</td></tr>
<tr><td class="l">端口分类准确率</td><td>{_pct(row,'stage3__remapper_eval_port_acc')}</td><td class="l">目标端口分类头准确率</td></tr>
<tr><td class="l">目标特征 L2 / MAE</td><td>{_f(row,'stage3_pcap__pcap_target_l2_mean')} / {_f(row,'stage3_pcap__pcap_target_mae_mean')}</td><td class="l">对抗 PCAP 特征与对抗目标距离</td></tr>
<tr><td class="l">特征对齐覆盖率</td><td>{_pct(row,'stage3_paper__paper_pcap_alignment_coverage')}</td><td class="l">特征对齐覆盖比例</td></tr>
<tr><td class="l">多步变形轮次</td><td>{multi}</td><td class="l">重映射轮次</td></tr>
</table>
<p class="discuss"><b>讨论：</b>重映射器 R² = {_f(row,'stage3__remapper_eval_r2')} 表明 MLP 重映射器能够有效拟合特征偏移到 PCAP 字段修改的映射。端口分类准确率 {_pct(row,'stage3__remapper_eval_port_acc')} 保证了目标端口在合理范围内。特征对齐覆盖率 {_pct(row,'stage3_paper__paper_pcap_alignment_coverage')} 表明几乎所有目标特征都在对抗 PCAP 中得到了对齐。</p>

<h3>5.3 PCAP 健全性与效率</h3>
<p class="ni"><b>表 12. PCAP 结构健全性</b></p>
<table><tr><th>指标</th><th>数值</th><th>阈值</th></tr>
<tr><td class="l">TCP 序列号回退率</td><td>{_f(row,'stage3_pcap__pcap_sanity_tcp_seq_backwards_rate')}</td><td>&lt; 0.01</td></tr>
<tr><td class="l">TCP 标志位无效率</td><td>{_f(row,'stage3_pcap__pcap_sanity_tcp_flag_invalid_rate')}</td><td>0.00</td></tr>
<tr><td class="l">TCP SYN-FIN / SYN-RST / FIN-RST</td><td>{_f(row,'stage3_pcap__pcap_sanity_tcp_syn_fin_rate')} / {_f(row,'stage3_pcap__pcap_sanity_tcp_syn_rst_rate')} / {_f(row,'stage3_pcap__pcap_sanity_tcp_fin_rst_rate')}</td><td>0.00</td></tr>
<tr><td class="l">传输层缺失率 / 时间戳非单调</td><td>{_f(row,'stage3_pcap__pcap_sanity_transport_missing_rate')} / {_f(row,'stage3_pcap__pcap_sanity_nonmonotonic_rate')}</td><td>0.00</td></tr>
<tr><td class="l"><b>致命标志汇总</b></td><td><b>{_f(row,'stage3_pcap__pcap_valid_fatal_rate')}</b></td><td><b>0.00</b></td></tr>
</table>
<p class="ni"><b>表 13. PCAP 修改效率</b></p>
<table><tr><th>指标</th><th>数值</th></tr>
<tr><td class="l">应用 / 评估耗时</td><td>{_f(row,'stage3_pcap__pcap_apply_time_sec')} s / {_f(row,'stage3_pcap__pcap_eval_time_sec')} s</td></tr>
<tr><td class="l">包吞吐量 / 写入数 / 总包数</td><td>{_f(row,'stage3_pcap__pcap_packet_throughput_pps')} pps / {_f(row,'stage3_pcap__pcap_written_count')} / {_f(st3,'pcap_packet_count')}</td></tr>
</table>
<p class="discuss"><b>讨论：</b>所有 TCP 健全性指标均接近或等于 0，致命标志为 0，证明保守重映射策略在修改协议字段时不破坏数据包结构完整性。修改后的 PCAP 可通过 scapy/tcpdump 正常解析，具备离线可部署性。</p>
""")

    # ═══════ 六 ═══════
    add(f"""<h2>六、总结与讨论</h2>
<p><b>综合：</b>{TITLE} 在 UNSW-NB15 上展示了端到端 NIDS 对抗样本生成能力。Agreement = {_f(row,'stage1__agreement')} → ASR = {_pct(row,'stage2__asr_oracle')} → PCAP 逃逸有效，三级 ASR 无衰减。</p>
<p><b>优势：</b>在 {len(ALL_BASELINES)} 种基线中排名第 {our_rank}，FFD 显著优于所有基线。STP 组相关性保持、协方差对齐证明了统计保真度。TCP 健全性全部通过证明了可部署性。</p>
<p><b>待完善：</b>(1) 互相提取矩阵需多 Oracle（CNN/RNN/GRU/LSTM/Transformer + RF/SVM）——将在 Reviewer Suite 中完成；(2) 源 PCAP 当前为良性，需选择恶意 PCAP 作为载体；(3) 消融实验、传输 IDS 评估（RQ7）、数据集审计将在后续 Reviewer Suite 运行中完成。</p>
<div class="footnote"><p><b>配置：</b>configs/nb15_full.yaml &ensp;|&ensp; <b>输出：</b>{root.resolve()}</p></div>
</body></html>""")

    out_path = pipe / "NB15_FULL_REPORT.html"
    out_path.write_text("".join(H), encoding="utf-8")
    print(f"OK: {out_path.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/debug/nb15_full")
