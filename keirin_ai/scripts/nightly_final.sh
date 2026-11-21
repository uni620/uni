#!/bin/bash
# 毎晩（23:40〜翌7:30）: 前日までのレースの確定オッズ（4券種）と3連単の確定オッズを Kドリームスから取り込む。
# 締切前オッズの記録（odds_watch.py）とは同時に動かさない（3連単のスクリプトは自分でも待つ）。間隔は3秒以上。
cd "$(dirname "$0")/.."
export TZ=Asia/Tokyo
python3 scripts/kd_final_odds.py 1800
python3 scripts/kd_final_trifecta.py "${1:-5000}"
