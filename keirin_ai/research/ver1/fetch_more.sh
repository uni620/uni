#!/bin/bash
# 過去データを新しい日から順にさかのぼって取る。保存済みのページは取り直さない。
# 毎朝の本番（7:41）と重ならないよう、7:15 になったら止める。
cd /home/user/uni/keirin_ai
export TZ=Asia/Tokyo
now=$(date +%s); stop=$(date -d "today 07:15" +%s); [ "$now" -ge "$stop" ] && stop=$(date -d "tomorrow 07:15" +%s)
h=$(date +%H%M); if [ "$h" -ge 0715 ] && [ "$h" -lt 0830 ]; then echo "本番の時間帯なので取得しない"; exit 0; fi
lim=$(( stop - now )); [ "$lim" -gt 7000 ] && lim=7000
echo "--- start $(date) limit ${lim}s" >> "$1"
timeout "$lim" python3 -m keirin fetch --from "${2:-2026-01-01}" --to 2026-08-19 2>&1 | grep -v "INFO\] 20" >> "$1"
echo "--- end $(date)" >> "$1"
