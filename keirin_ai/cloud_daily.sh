#!/bin/bash
# クラウド（Linux）で毎朝1回動かす本番の流れ。時刻はすべて日本時間で扱う。
#   1. 昨日・一昨日の出走表と結果を取り込む（取りこぼしがあっても2日分はさかのぼって埋める）
#   2. 仮想購入を精算 → 全データで AI を学び直す → Excel を更新
#   3. 今日のレースのうち、まだ発走していないものを全部予想して仮想購入を記録する
#      （判断はこの時点の情報だけ。オッズは使わない。あとで欠車が出た買い目は返金扱い）
set -euo pipefail
cd "$(dirname "$0")"
export TZ=Asia/Tokyo

python3 -m keirin fetch --days 2
python3 -m keirin daily --date "$(date -d yesterday +%F)"
python3 -m keirin predict --all
