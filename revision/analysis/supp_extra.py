import numpy as np, pandas as pd
src=open('revision_analysis.py').read().split('# ----------------------------------------------------------------- 0.')[0]
exec(src)
tf=file_table('test')
tt=pd.read_csv(REPO / 'outputs' / 'phash_clean_sensitivity' / 'affected_test_images_train_test.csv')
an=pd.read_csv(AFFECTED)
col=lambda d:[c for c in d.columns if 'path' in c][0]
m_tt=~tf.relative_path.isin(set(tt[col(tt)])).values; m_any=~tf.relative_path.isin(set(an[col(an)])).values
rows=[]
for lab,exp in [('CE',FACT['CE']),('WCE + sampler',FACT['WCE + sampler']),('Moderate ordinal',ORD)]:
    for s in SEEDS5:
        y,P=load(exp,s); p=P.argmax(1)
        for sub,m in [('Official (n=1656)',np.ones(len(y),bool)),(f'Train-test clean (n={m_tt.sum()})',m_tt),(f'Any-cross-split clean (n={m_any.sum()})',m_any)]:
            rows.append({'model':lab,'subset':sub,'seed':s,**metrics(y[m],p[m])})
d=pd.DataFrame(rows); g=d.groupby(['model','subset'],sort=False)[['qwk','acc','macro_f1','g1_recall']].agg(['mean','std'])
g.to_csv(str(OUT / 'tables')+'/official_models_on_phash_clean_test_subsets.csv'); print(g.round(4).to_string())
