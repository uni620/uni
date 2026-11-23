"""ガールズ頭AI v1（2026-11-23 固定）。オッズ・本紙は使わない。

ガールズケイリン専用。出走表の数字（競走得点・勝率・連対率・決まり手・S/B）とその順位、
直近の成績（直近5走の着順割合・上がり順位、直近10走の勝率）から、
頭（lambdarank で並べた1位）と1着の確率（binary、レース内で合計1）を出す。相手は同じ並びの2〜4位。
記録（頭の確率 0.8 以上のレースだけ）: ワイド3点（頭-相手3人）と 2車単3点（頭→相手3人）。
検証（2026-04〜11、各月その月より前で学習、1,495R）: 頭の確率80%以上 586R で 頭の的中 84.8%、
ワイド3点 的中 約95%（回収率 約82%）、2車単3点 的中 72.5%（回収率 77%）。回収率は100%未満（的中を優先した形）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GIRLS_HEAD = "ガールズ頭AI v1"
BET_G_WIDE = "ガールズ頭AI v1・ワイド3点（確率80%以上）"
BET_G_2T = "ガールズ頭AI v1・2車単3点（確率80%以上）"
MIN_PR = 0.8
QUERY = """SELECT e.race_id, e.car_no, e.rider_id, e.score, e.finish, e.scratched, e.win_rate, e.ren2, e.ren3,
  e.km_nige, e.km_makuri, e.km_sashi, e.km_mark, e.sb_s, e.sb_h, e.sb_b, e.n1, e.n2, e.n3, e.nout, e.gear, e.age,
  r.date, r.bank_len, i.agari
  FROM entries e JOIN races r USING(race_id) LEFT JOIN result_info i USING(race_id, car_no)
  WHERE r.race_class LIKE '%ガールズ%'"""
NUM = ['score', 'finish', 'win_rate', 'ren2', 'ren3', 'km_nige', 'km_makuri', 'km_sashi', 'km_mark',
       'sb_s', 'sb_h', 'sb_b', 'n1', 'n2', 'n3', 'nout', 'gear', 'age', 'agari', 'bank_len']
LR = dict(objective='lambdarank', learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, verbose=-1, eval_at=[1], lambdarank_truncation_level=3)
BIN = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def build(con, today: str) -> tuple[pd.DataFrame, list[str]]:
    E = pd.read_sql(QUERY, con)
    E = E[E.date <= today].copy()
    for k in NUM:
        E[k] = pd.to_numeric(E[k], errors='coerce')
    E['dt'] = pd.to_datetime(E.date)
    E['n'] = E.groupby('race_id').car_no.transform('size')
    st = E[['n1', 'n2', 'n3', 'nout']].sum(axis=1).replace(0, np.nan)
    km = E[['km_nige', 'km_makuri', 'km_sashi', 'km_mark']].sum(axis=1).replace(0, np.nan)
    E['B率'] = E.sb_b / st; E['S率'] = E.sb_s / st
    E['逃率'] = E.km_nige / km; E['捲率'] = E.km_makuri / km; E['差率'] = E.km_sashi / km
    for k in ['score', 'win_rate', 'ren2', 'ren3', 'B率', 'S率', '逃率', '捲率']:
        E[k + '_順位'] = E.groupby('race_id')[k].rank(ascending=False)
        E[k + '_差'] = E[k] - E.groupby('race_id')[k].transform('max')
    E['得点2位との差'] = E.score - E.groupby('race_id').score.transform(lambda s: s.nlargest(2).iloc[-1])
    E = E.sort_values(['rider_id', 'dt', 'race_id'])
    E['fin_pct'] = E.finish / E.n
    E['win'] = (E.finish == 1).astype(float)
    E.loc[E.finish.isna(), 'win'] = np.nan
    E['agari_rk'] = E.groupby('race_id').agari.rank(pct=True)
    g = E.groupby('rider_id')
    for col, new, w in (('fin_pct', '近5着', 5), ('win', '近10勝率', 10), ('agari_rk', '近5上がり', 5)):
        s = g[col].shift(1)
        E[new] = s.groupby(E.rider_id).transform(lambda v: v.rolling(w, min_periods=2).mean())
    for k in ['近5着', '近10勝率', '近5上がり']:
        E[k + '_順位'] = E.groupby('race_id')[k].rank(ascending=(k != '近10勝率'))
    F = [c for c in E.columns if c.endswith(('_順位', '_差'))] + \
        ['score', 'win_rate', 'ren2', 'ren3', 'B率', 'S率', '逃率', '捲率', '差率', 'gear', 'age', 'n', 'car_no',
         'bank_len', '得点2位との差', '近5着', '近10勝率', '近5上がり']
    return E.sort_values(['race_id', 'car_no']).reset_index(drop=True), F


def predict(con, today: str) -> tuple[pd.DataFrame, list[dict]]:
    """→ (race_id, car_no, pr) と、頭の確率 MIN_PR 以上のレースの買い目。"""
    import lightgbm as lgb
    E, F = build(con, today)
    scr = set(E[(E.scratched == 1) & (E.date < today)].race_id)
    hist = E[(E.date < today) & ~E.race_id.isin(scr)]
    hist = hist[hist.groupby('race_id').finish.transform(lambda s: (s == 1).sum() == 1)]
    y = E[E.date == today].copy()
    if y.empty or hist.empty:
        return pd.DataFrame(columns=['race_id', 'car_no', 'pr']), []
    lab = np.select([hist.finish == 1, hist.finish == 2, hist.finish == 3], [3, 2, 1], 0)
    d = lgb.Dataset(hist[F], lab)
    d.set_group(hist.groupby('race_id', sort=False).size().values)
    y['s'] = sum(lgb.train(dict(LR, seed=s), d, 400).predict(y[F]) for s in (1, 2))
    y['pr'] = sum(lgb.train(dict(BIN, seed=s), lgb.Dataset(hist[F], hist.win), 400).predict(y[F]) for s in (1, 2))
    y['pr'] = y.pr / y.groupby('race_id').pr.transform('sum')
    picks = []
    for rid, g in y.groupby('race_id'):
        g = g.sort_values('s', ascending=False)
        o = g.car_no.astype(int).tolist()
        h, pr = o[0], float(g.pr.max())
        if pr < MIN_PR or len(o) < 4:
            continue
        picks.append(dict(race_id=rid, head=h, conf=pr, sec=o[1:4],
                          bets={BET_G_WIDE: [('ワイド', '-'.join(map(str, sorted((h, x)))), pr) for x in o[1:4]],
                                BET_G_2T: [('2車単', f'{h}>{x}', pr) for x in o[1:4]]}))
    return y[['race_id', 'car_no', 'pr']], picks
