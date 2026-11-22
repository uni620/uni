"""頭AI v1 ＋ 相手AI（展開型）v1（2026-11-22 固定）。オッズ・本紙は使わない。

頭AI v1: その選手が1着になる確率（LightGBM binary、レース内で合計1）。
  材料 = 展開AI v1 の主導権確率（pB）とライン、選手の状況別成績（B取り時・先頭・番手・単騎、過去120日）、
  本当の調子（直近5走の上がり順位・勝者からの着差・着順）、90日の ちぎられ/ちぎり/まくり、競走得点・勝率、連係・対戦。
相手AI（展開型）v1: 頭を出発点に、段階1 = 頭が勝つときに誰が B を取るか、段階2 = その展開で誰が2着・3着か。
  B の候補上位3人を、その確率で重み付けして合わせる。
買い目ごとの見込み確率: 2車単 = 頭の確率×2着の確率、3連単 = 頭×2着×3着/(1−その2着の3着確率)。
記録する買い方（頭AIの自信 = 頭の確率が 0.6 以上のレースだけ）:
  2車単3点 = 頭 → 2着候補上位3人 / 3連単9点 = 頭 → 2着候補上位3人 → 3着候補上位3人（2着と同じ選手は除く）
検証（2026-04〜11、各月その月より前だけで学習）: 自信60%以上 1,270R で頭の的中 約78%、
  2車単3点 的中66.8%・回収率75%、3連単9点 的中56.4%・回収率80%。回収率は100%未満（的中を優先した形）。
  さらに締切8分前に、3連単9点のうち 見込み確率×8分前オッズ が EV_MIN を超える目だけを別に記録する（BET_3T_EV）。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from keirin.tenkai import A as TENKAI_A, P as TENKAI_P, QUERY, features, stage2_features

HEAD_NAME = "頭AI v1"
BET_2T = "頭AI v1＋相手AI展開型v1・2車単3点（自信60%以上）"
BET_3T = "頭AI v1＋相手AI展開型v1・3連単9点（自信60%以上）"
BET_3T_EV = "頭AI v1＋相手AI展開型v1・3連単9点のうち8分前EV1.2超"
EV_MIN = 1.2   # 8分前の3連単オッズ×見込み確率 がこれを超える目だけ残す（2026-11-22 決定。確定オッズでの検証は EV1.2以上で回収率96%・95%幅62〜138%）
MIN_CONF = 0.6

FEAT = ['B_1着率', 'B_3着内率', 'B_回数', '先_1着率', '先_3着内率', '先_回数', '番_1着率', '番_3着内率', '番_回数', '単_3着内率',
        '近5_agari_pct', '近5_behind', '近5_fin_pct', '内容−着順', '今回の位置の3着内率', 'pB', '自分のラインのpB', '自分の先頭のpB',
        '他ラインの最大pB', '競り合い', '主導権×B時1着率', '先頭', '番手', '3番手+', '単騎', 'ライン人数', 'n', 'ライン数', '先行候補数',
        '逃率', '捲率', '差率', 'マ率', 'B率', 'S率', 'ちぎられ率', 'ちぎり率', 'まくり率', 'まくり回数', '上がり差', 'B時の平均着順',
        'score', '得点差', '得点順位', '得点1位との差', '自ライン平均得点', '自ライン得点順位', 'win_rate', 'ren2', 'ren3', '勝率順位',
        '連係回数', '連係時の先頭3着内率', '連係時のワンツー率', '対戦数', '相手への先着率', '先着率順位', '番手との連係回数']
REL = ('と同ライン', 'の直後', 'の直前', 'の2つ後ろ', 'が先頭', 'が番手', 'が単騎', 'との得点差')
RH = [f'頭{s}' for s in REL]
RB = [f'B{s}' for s in REL]
PBIN = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, verbose=-1)
MARGIN = {'同着': 0, '微差': .05, 'タイヤ': .05, '1/8車輪': .06, '1/4車輪': .1, '1/2車輪': .2, '3/4車輪': .3, '1車輪': .4,
          '1/2車身': .5, '3/4車身': .75, '1車身': 1, '1車身1/2': 1.5, '大差': 10}


def _margin(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0.0
    s = str(s)
    if s in MARGIN:
        return MARGIN[s]
    if s.endswith('車身') and s[:-2].isdigit():
        return float(s[:-2])
    return np.nan


def _rolling(E: pd.DataFrame, cols: list[str], days: str) -> pd.DataFrame:
    """選手×日の合計を、その日より前の days 日分だけ足す（当日は含めない）。"""
    g = E.groupby(['rider_id', 'dt'])[cols].sum().reset_index().sort_values(['rider_id', 'dt'])
    parts = []
    for rid, x in g.groupby('rider_id'):
        y = x.set_index('dt')[cols].rolling(days, closed='left').sum()
        y['rider_id'] = rid
        parts.append(y.reset_index())
    return pd.concat(parts)


def _pair_h2h(E: pd.DataFrame) -> pd.DataFrame:
    """連係（同じラインで前後に並んだ過去）と対戦（今回の相手への過去の先着率）。その日より前だけ。"""
    D = E.sort_values(['date', 'race_id', 'car_no'])
    pair = defaultdict(lambda: [0, 0, 0])
    h2h = defaultdict(lambda: [0, 0])
    feat = {}
    for _, G in D.groupby('date', sort=True):
        races = [g.to_dict('records') for _, g in G.groupby('race_id')]
        for rows in races:
            by_lk = defaultdict(list)
            for r in rows:
                by_lk[r['lk']].append(r)
            for r in rows:
                pn = p3 = p12 = 0
                for m in by_lk[r['lk']]:
                    if m['car_no'] == r['car_no'] or abs((m['line_pos'] or 0) - (r['line_pos'] or 0)) != 1:
                        continue
                    a, b = (m['rider_id'], r['rider_id']) if (m['line_pos'] or 0) < (r['line_pos'] or 0) else (r['rider_id'], m['rider_id'])
                    v = pair[(a, b)]; pn += v[0]; p3 += v[1]; p12 += v[2]
                w = n = 0
                for o in rows:
                    if o['car_no'] == r['car_no']:
                        continue
                    a, b = r['rider_id'], o['rider_id']
                    if a < b:
                        v = h2h[(a, b)]; w += v[0]; n += v[1]
                    else:
                        v = h2h[(b, a)]; w += v[1] - v[0]; n += v[1]
                feat[(r['race_id'], r['car_no'])] = (pn, (p3 + 0.5) / (pn + 1), (p12 + 0.2) / (pn + 1), n, (w + 1) / (n + 2))
        for rows in races:
            rows = [r for r in rows if not np.isnan(r['finish']) and r['scratched'] != 1]
            for r in rows:
                for o in rows:
                    if r['lk'] == o['lk'] and (r['line_pos'] or 0) + 1 == (o['line_pos'] or 0):
                        v = pair[(r['rider_id'], o['rider_id'])]
                        v[0] += 1; v[1] += r['finish'] <= 3; v[2] += (r['finish'] <= 2 and o['finish'] <= 2)
                    if r['rider_id'] < o['rider_id']:
                        v = h2h[(r['rider_id'], o['rider_id'])]; v[0] += r['finish'] < o['finish']; v[1] += 1
    return pd.DataFrame([k + v for k, v in feat.items()],
                        columns=['race_id', 'car_no', '連係回数', '連係時の先頭3着内率', '連係時のワンツー率', '対戦数', '相手への先着率'])


def build(con, today: str) -> pd.DataFrame:
    """全レース（today まで）の特徴を作る。today の行は結果なし。"""
    import lightgbm as lgb
    q = QUERY.replace("r.date, r.race_class, i.sb",
                      "r.date, r.race_class, i.sb, i.agari, i.margin, e.rider_id, e.score, e.win_rate, e.ren2, e.ren3, e.kimarite")
    E = features(pd.read_sql(q, con))
    E = E[E.date <= today].copy()
    E['finish'] = pd.to_numeric(E.finish, errors='coerce')
    for k in ('score', 'win_rate', 'ren2', 'ren3'):
        E[k] = pd.to_numeric(E[k], errors='coerce')
    scr = set(E[(E.scratched == 1) & (E.date < today)].race_id)
    E = E[~E.race_id.isin(scr)].copy()
    E['dt'] = pd.to_datetime(E.date)
    E['m1'] = E.margin.map(_margin)
    E = E.sort_values(['race_id', 'finish'])
    E['behind'] = E.groupby('race_id').m1.cumsum().clip(upper=10)
    E.loc[E.finish.isna(), 'behind'] = np.nan
    E['agari'] = pd.to_numeric(E.agari, errors='coerce')
    E['agari_pct'] = E.groupby('race_id').agari.rank(pct=True)
    E['fin_pct'] = E.finish / E.n
    E['gotB'] = E.sb.astype(str).str.contains('B').astype(float)
    E['top3'] = (E.finish <= 3).astype(float)
    E['win'] = (E.finish == 1).astype(float)
    # 状況別の成績（120日）
    for k, cond in (('B', E.gotB == 1), ('先', E.先頭 == 1), ('番', E.番手 == 1), ('単', E.単騎 == 1)):
        E[f'n_{k}'] = cond.astype(float)
        E[f'w_{k}'] = (cond & (E.win == 1)).astype(float)
        E[f't_{k}'] = (cond & (E.top3 == 1)).astype(float)
    ev = [f'{a}_{k}' for k in 'B先番単' for a in 'nwt']
    E = E.merge(_rolling(E, ev, '120D'), on=['rider_id', 'dt'], how='left', suffixes=('', '_h'))
    for k in 'B先番単':
        n = E[f'n_{k}_h']
        E[f'{k}_1着率'] = (E[f'w_{k}_h'] + 0.1) / (n + 1)
        E[f'{k}_3着内率'] = (E[f't_{k}_h'] + 0.33) / (n + 1)
        E[f'{k}_回数'] = n
    # 直近5走の内容
    E = E.sort_values(['rider_id', 'dt', 'race_id'])
    for col in ('agari_pct', 'behind', 'fin_pct'):
        s = E.groupby('rider_id')[col].shift(1)
        E[f'近5_{col}'] = s.groupby(E.rider_id).transform(lambda v: v.rolling(5, min_periods=2).mean())
    E['内容−着順'] = E['近5_fin_pct'] - E['近5_agari_pct']
    E['今回の位置の3着内率'] = np.select([E.先頭 == 1, E.番手 == 1, E.単騎 == 1], [E['先_3着内率'], E['番_3着内率'], E['単_3着内率']], np.nan)
    # 90日の ちぎられ/ちぎり/まくり（展開AI v2 と同じ）
    ld = E[E.先頭 == 1][['lk', 'finish']].rename(columns={'finish': 'Lf'})
    bn = E[E.番手 == 1][['lk', 'finish']].rename(columns={'finish': 'Bf'})
    E = E.merge(ld, on='lk', how='left').merge(bn, on='lk', how='left')
    fol = (E.番手 == 1) | (E['3番手+'] == 1)
    E['ev_follow'] = (fol & (E.Lf <= 3)).astype(float)
    E['ev_chigirare'] = (fol & (E.Lf <= 3) & (E.finish >= E.Lf + 3)).astype(float)
    E['ev_lead'] = ((E.先頭 == 1) & (E.finish <= 3) & E.Bf.notna()).astype(float)
    E['ev_chigiri'] = ((E.先頭 == 1) & (E.finish <= 3) & (E.Bf >= E.finish + 3)).astype(float)
    E['ev_makuri'] = (E.kimarite.astype(str).str.contains('捲') & (E.finish <= 2)).astype(float)
    E['ev_race'] = E.finish.notna().astype(float)
    E['ev_agari'] = E.agari - E.groupby('race_id').agari.transform('median')
    E['ev_agari_n'] = E.ev_agari.notna().astype(float); E['ev_agari'] = E.ev_agari.fillna(0)
    E['ev_Bfin'] = E.gotB * E.finish.fillna(0)
    ev2 = ['ev_follow', 'ev_chigirare', 'ev_lead', 'ev_chigiri', 'ev_makuri', 'ev_race', 'ev_agari', 'ev_agari_n', 'gotB', 'ev_Bfin']
    E = E.merge(_rolling(E, ev2, '90D'), on=['rider_id', 'dt'], how='left', suffixes=('', '_90'))
    f = lambda a, b: E[a + '_90'] / E[b + '_90'].replace(0, np.nan)  # noqa: E731
    E['ちぎられ率'] = f('ev_chigirare', 'ev_follow'); E['ちぎり率'] = f('ev_chigiri', 'ev_lead')
    E['まくり率'] = f('ev_makuri', 'ev_race'); E['まくり回数'] = E['ev_makuri_90']
    E['上がり差'] = f('ev_agari', 'ev_agari_n'); E['B時の平均着順'] = f('ev_Bfin', 'gotB')
    # 展開AI v1 段階1（主導権の確率 pB）: 過去は2つに分けて交互、today は過去全部で
    hist = (E.date < today) & (E.groupby('race_id').gotB.transform('sum') == 1)
    E['pB'] = np.nan
    H = E[hist].sort_values('date')
    half = E.date < H.date.iloc[len(H) // 2]
    for a, b in ((hist & half, hist & ~half), (hist & ~half, hist & half)):
        m = lgb.train(TENKAI_P, lgb.Dataset(E.loc[a, TENKAI_A], E.loc[a, 'gotB']), 300)
        E.loc[b, 'pB'] = m.predict(E.loc[b, TENKAI_A])
    m = lgb.train(TENKAI_P, lgb.Dataset(E.loc[hist, TENKAI_A], E.loc[hist, 'gotB']), 300)
    rest = E.pB.isna()
    E.loc[rest, 'pB'] = m.predict(E.loc[rest, TENKAI_A])
    E = stage2_features(E)
    E['主導権×B時1着率'] = E.pB * E['B_1着率']
    # 強さ
    E['得点差'] = E.score - E.groupby('race_id').score.transform('mean')
    E['得点順位'] = E.groupby('race_id').score.rank(ascending=False)
    E['得点1位との差'] = E.score - E.groupby('race_id').score.transform('max')
    E['自ライン平均得点'] = E.groupby('lk').score.transform('mean')
    E['自ライン得点順位'] = E.groupby('race_id').自ライン平均得点.rank(ascending=False, method='dense')
    E['勝率順位'] = E.groupby('race_id').win_rate.rank(ascending=False)
    # 連係・対戦
    E = E.merge(_pair_h2h(E), on=['race_id', 'car_no'], how='left')
    E['先着率順位'] = E.groupby('race_id').相手への先着率.rank(ascending=False)
    E['番手との連係回数'] = E.lk.map(E[E.line_pos == 2].drop_duplicates('lk').set_index('lk').連係回数).where(E.line_pos == 1)
    return E.sort_values(['date', 'race_id', 'car_no']).reset_index(drop=True)


def _relto(X: pd.DataFrame, key: pd.DataFrame, ref: pd.Series, pfx: str) -> pd.DataFrame:
    r = key.loc[list(zip(ref.index, ref.values))].reset_index().set_index('race_id')
    same = X.lk == X.race_id.map(r.lk)
    d = X.line_pos - X.race_id.map(r.line_pos)
    X[pfx + 'と同ライン'] = same.astype(float); X[pfx + 'の直後'] = (same & (d == 1)).astype(float)
    X[pfx + 'の直前'] = (same & (d == -1)).astype(float); X[pfx + 'の2つ後ろ'] = (same & (d == 2)).astype(float)
    X[pfx + 'が先頭'] = X.race_id.map(r.先頭); X[pfx + 'が番手'] = X.race_id.map(r.番手); X[pfx + 'が単騎'] = X.race_id.map(r.単騎)
    X[pfx + 'との得点差'] = X.score - X.race_id.map(r.score)
    return X


def predict(con, today: str) -> tuple[pd.DataFrame, list[dict]]:
    """→ (頭AIの確率: race_id, car_no, pr) と、自信 MIN_CONF 以上のレースの買い目リスト。"""
    import lightgbm as lgb
    E = build(con, today)
    hist = E[(E.date < today) & (E.groupby('race_id').win.transform('sum') == 1)]
    hist = hist[hist.groupby('race_id').gotB.transform('sum') == 1]
    y = E[E.date == today].copy()
    if y.empty or hist.empty:
        return pd.DataFrame(columns=['race_id', 'car_no', 'pr']), []
    y['pr'] = sum(lgb.train(dict(PBIN, seed=s), lgb.Dataset(hist[FEAT], hist.win), 500).predict(y[FEAT]) for s in (1, 2))
    y['pr'] = y.pr / y.groupby('race_id').pr.transform('sum')
    key = E.set_index(['race_id', 'car_no'])
    hw = hist[hist.win == 1].set_index('race_id').car_no
    bw = hist[hist.gotB == 1].set_index('race_id').car_no
    X1 = _relto(hist.copy(), key, hw, '頭')
    m1 = lgb.train(PBIN, lgb.Dataset(X1[FEAT + RH], X1.gotB), 400)
    X2 = _relto(_relto(hist[hist.win == 0].copy(), key, hw, '頭'), key, bw, 'B')
    m2 = lgb.train(PBIN, lgb.Dataset(X2[FEAT + RH + RB], (X2.finish == 2).astype(int)), 400)
    m3 = lgb.train(PBIN, lgb.Dataset(X2[FEAT + RH + RB], (X2.finish == 3).astype(int)), 400)
    hh = y.loc[y.groupby('race_id').pr.idxmax()].set_index('race_id')
    hh = hh[hh.pr >= MIN_CONF]
    picks = []
    if hh.empty:
        return y[['race_id', 'car_no', 'pr']], picks
    Y = y[y.race_id.isin(hh.index)]
    Y1 = _relto(Y.copy(), key, hh.car_no, '頭')
    Y1['pb'] = m1.predict(Y1[FEAT + RH]); Y1['pb'] /= Y1.groupby('race_id').pb.transform('sum')
    cand = Y1.sort_values(['race_id', 'pb'], ascending=[True, False]).groupby('race_id').head(3)
    others = Y[Y.car_no != Y.race_id.map(hh.car_no)]
    parts = []
    for k in range(3):
        bk = cand.groupby('race_id').nth(k).set_index('race_id')
        Z = _relto(_relto(others[others.race_id.isin(bk.index)].copy(), key, hh.car_no, '頭'), key, bk.car_no, 'B')
        Z['w'] = Z.race_id.map(bk.pb)
        Z['p2'] = m2.predict(Z[FEAT + RH + RB]); Z['p3'] = m3.predict(Z[FEAT + RH + RB])
        for col in ('p2', 'p3'):
            Z[col] = Z[col] / Z.groupby('race_id')[col].transform('sum') * Z.w
        parts.append(Z[['race_id', 'car_no', 'p2', 'p3']])
    Z = pd.concat(parts).groupby(['race_id', 'car_no'])[['p2', 'p3']].sum().reset_index()
    for rid, h in hh.iterrows():
        z = Z[Z.race_id == rid]
        if z.empty:
            continue
        hd = int(h.car_no)
        q2 = (z.set_index('car_no').p2 / z.p2.sum()); q3 = (z.set_index('car_no').p3 / z.p3.sum())
        q2.index = q2.index.astype(int); q3.index = q3.index.astype(int)
        s2 = q2.sort_values(ascending=False).index.tolist()
        s3 = (q2 + q3).sort_values(ascending=False).index.tolist()
        pr = float(h.pr)
        b1 = int(cand[cand.race_id == rid].car_no.iloc[0])
        picks.append(dict(race_id=rid, head=hd, conf=float(h.pr), b=b1, sec=s2[:3],
                          bets={BET_2T: [('2車単', f'{hd}>{u}', pr * q2[u]) for u in s2[:3]],
                                BET_3T: [('3連単', f'{hd}>{u}>{v}', pr * q2[u] * q3[v] / max(1 - q3[u], 1e-6))
                                         for u in s2[:3] for v in [t for t in s3 if t != u][:3]]}))
    return y[['race_id', 'car_no', 'pr']], picks
