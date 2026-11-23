import sys, json, pandas as pd
sys.path.insert(0,'/home/user/uni/keirin_ai'); sys.path.insert(0,'/home/user/uni/keirin_ai/scripts')
from fit_line_combos import load
from keirin.line_combos import line_combo_probs
pr=json.load(open('line_fit_1015.json'))['params']
rows=[]
for r in load('wf_v2.pkl','LightGBM+競走得点'):
    cp=line_combo_probs(r['cars'],r['p'],r['lines'],pr)
    for bt,dd in cp.items():
        for cb,v in dd.items(): rows.append((r['race_id'],bt,cb,v))
pd.DataFrame(rows,columns=['race_id','bt','combo','pl']).to_pickle('line_probs_all.pkl')
print(len(rows))
