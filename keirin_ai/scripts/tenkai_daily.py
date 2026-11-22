"""展開AI v1 で今日のレースを予想して記録する（predictions_m の method='展開AI v1'＝1着確率、'展開AI v1・主導権'＝Bを取る確率）。
毎朝の処理（cloud_daily.sh）の最後に動かす。通信はしない（DB にある出走表だけ使う）。"""
import sys
from datetime import date
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keirin.db import DB  # noqa: E402
from keirin.tenkai import train_and_predict  # noqa: E402
day = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
db = DB(str(ROOT / "data" / "keirin.sqlite"))
out = train_and_predict(db.con, day)
for rid, g in out.groupby("race_id"):
    db.save_method_preds(rid, "展開AI v1", g.car_no.astype(int).tolist(), g.p_win.tolist())
    db.save_method_preds(rid, "展開AI v1・主導権", g.car_no.astype(int).tolist(), g.pB.tolist())
db.commit()
print(f"展開AI v1: {out.race_id.nunique()} レースを予想して記録しました（{day}）")
