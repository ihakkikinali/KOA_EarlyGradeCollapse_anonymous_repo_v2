import pandas as pd, numpy as np, json
from pathlib import Path
T=str(Path(__file__).resolve().parent/'results'/'tables')+'/'
LT=Path(__file__).resolve().parent/'results'/'latex'; LT.mkdir(exist_ok=True); out=[]
f3=lambda x:f"{x:.3f}"
def pm(m,s): return f"${m:.3f}\\pm{s:.3f}$"
# S2 per-seed
s=pd.read_csv(T+'per_seed_all_arms.csv')
name={'CE':'CE','WCE only':'WCE only','Sampler only':'Sampler only','WCE + sampler':'WCE + sampler','Ordinal (moderate)':'Moderate ordinal',
'pHash-clean WCE + sampler':'pHash-clean WCE + sampler','pHash-clean ordinal':'pHash-clean ordinal','Ordinal λ 0.05/0.02':'Ordinal $\\lambda$ 0.05/0.02','Ordinal λ 0.35/0.20':'Ordinal $\\lambda$ 0.35/0.20'}
rows=[]
for _,r in s.iterrows():
    rows.append(f"{name[r.arm]} & {r.seed} & {f3(r.val_score)} & {f3(r.qwk)} & {f3(r.acc)} & {f3(r.mae)} & {f3(r.macro_f1)} & {f3(r.g1_recall)} & {f3(r.g1_precision)} & {f3(r.g1_spec)} \\\\")
open(str(LT)+'/S2.tex','w').write("\n".join(rows))
# S3
d=pd.read_csv(T+'official_models_on_phash_clean_test_subsets.csv',header=[0,1],index_col=[0,1])
rows=[]
for (mod,sub),r in d.iterrows():
    rows.append(f"{mod} & {sub} & {pm(r[('qwk','mean')],r[('qwk','std')])} & {pm(r[('acc','mean')],r[('acc','std')])} & {pm(r[('macro_f1','mean')],r[('macro_f1','std')])} & {pm(r[('g1_recall','mean')],r[('g1_recall','std')])} \\\\")
open(str(LT)+'/S3.tex','w').write("\n".join(rows))
# S4 calibration
e=pd.read_csv(T+'ensemble5_metrics.csv').set_index('arm')
rows=[]
for a in ['CE','WCE only','Sampler only','WCE + sampler','Ordinal (moderate)']:
    x=s[s.arm==a]
    rows.append(f"{name[a]} & {pm(x.ece.mean(),x.ece.std())} & {pm(x.brier.mean(),x.brier.std())} & {f3(e.loc[a,'ece'])} & {f3(e.loc[a,'brier'])} \\\\")
open(str(LT)+'/S4.tex','w').write("\n".join(rows))
# S5 logit adjustment
l=pd.read_csv(T+'logit_adjustment_ce_per_seed.csv'); rows=[]
for tau,x in l.groupby('tau'):
    rows.append(f"{tau:.2f} & "+" & ".join(pm(x[k].mean(),x[k].std()) for k in ['qwk','acc','macro_f1','g1_recall','g1_precision','g1_spec'])+" \\\\")
ref=s[s.arm=='WCE only']; rows.append("\\midrule\nWCE only (reference) & "+" & ".join(pm(ref[k].mean(),ref[k].std()) for k in ['qwk','acc','macro_f1','g1_recall','g1_precision','g1_spec'])+" \\\\")
open(str(LT)+'/S5.tex','w').write("\n".join(rows))
# S6 bias audit
b=pd.read_csv(T+'validation_bias_audit.csv'); rows=[]
for _,r in b.iterrows():
    rows.append(f"{name[r.arm]} & {r.seed} & ({r.bias_0:+.2f}, {r.bias_1:+.2f}, {r.bias_2:+.2f}) & {r.val_delta_acc:+.4f} & {r.test_delta_acc:+.4f} & {r.val_delta_g1_recall:+.3f} & {r.test_delta_g1_recall:+.3f} \\\\")
open(str(LT)+'/S6.tex','w').write("\n".join(rows))
# S8 Grade-1 with CI
g=pd.read_csv(T+'grade1_metrics_with_ci.csv'); rows=[]
for a in ['CE','WCE only','Sampler only','WCE + sampler','Ordinal (moderate)']:
    x=g[g.arm==a].set_index('metric')
    rows.append(f"{name[a]} & "+" & ".join(f"{x.loc[k,'mean']:.3f} [{x.loc[k,'ci_lo']:.3f}, {x.loc[k,'ci_hi']:.3f}]" for k in ['g1_recall','g1_precision','g1_f1','g1_spec'])+" \\\\")
open(str(LT)+'/S8.tex','w').write("\n".join(rows))
# S9 confusion flows
c=json.load(open(T+'ensemble5_confusion.json')); rows=[]
for a in ['CE','WCE only','Sampler only','WCE + sampler','Ordinal (moderate)']:
    m=np.array(c[a]); rows.append(f"{name[a]} & {m[1,0]} & {m[1,1]} & {m[1,2]} & {m[1,3]+m[1,4]} & {m[0,1]} & {m[2,1]} & {m[:,1].sum()} \\\\")
open(str(LT)+'/S9.tex','w').write("\n".join(rows))
print('ok')
