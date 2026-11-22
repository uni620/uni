#!/bin/bash
# クラウド（Linux）で毎朝1回動かす本番の流れ。時刻はすべて日本時間で扱う。
#   1. まだ結果が無いレースの結果だけを取り込む（出走表は取り直さない。2026-11-22 閲覧回数を減らすため）
#   2. 仮想購入を精算 → 全データで AI を学び直す → Excel を更新
#   3. 今日のレースのうち、まだ発走していないものを全部予想して仮想購入を記録する
#      （判断はこの時点の情報だけ。オッズは使わない。あとで欠車が出た買い目は返金扱い）
set -euo pipefail
cd "$(dirname "$0")"
export TZ=Asia/Tokyo

python3 -m keirin daily --date "$(date -d yesterday +%F)"
python3 -m keirin predict --all
python3 -m keirin report
python3 scripts/tenkai_daily.py "$(date +%F)"
python3 scripts/head_daily.py "$(date +%F)"
