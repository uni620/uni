from hx_lib import *
E=pd.read_pickle(f'{S}/hx2.pkl'); C=pickle.load(open(f'{S}/hx_cols.pkl','rb'))
PFC,CC=C['PFC'],C['CC']; FEAT=[f for f in C['FEAT'] if f not in PFC]; base=FEAT+CC
for nm,p,r,lab in (
  ('葉7・ゆっくり',dict(num_leaves=7,learning_rate=0.02,min_data_in_leaf=300),800,'rank3'),
  ('葉15',dict(num_leaves=15,min_data_in_leaf=200),500,'rank3'),
  ('葉63',dict(num_leaves=63,min_data_in_leaf=50),300,'rank3'),
  ('特徴50%',dict(feature_fraction=0.5),400,'rank3'),
  ('ラベル=1着だけ',{},400,'win'),
  ('ラベル=4着まで',{},400,'rank5'),
  ('xendcg',dict(objective='rank_xendcg'),400,'rank3'),
  ('binary',dict(objective='binary'),400,'win'),
  ('lambdarank 800回 lr0.015',dict(learning_rate=0.015),800,'rank3'),
):
    print(nm,evaluate(E,base,p,r,lab=lab),flush=True)
# 小さい特徴だけ
small=['得点差','得点1位との差','得点順位','勝率順位','win_rate','自分のラインのpB','自分の先頭のpB','他ラインの最大pB','pB','主導権×B時1着率','今回の位置の3着内率','先頭','番手','単騎','3番手+','ライン人数','n']
print('特徴17個だけ',evaluate(E,small),flush=True)
