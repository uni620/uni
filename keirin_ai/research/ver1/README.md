# 予想ルール ver1 の検証スクリプト（2026-11-22 固定）

ver1 = 「オッズ1点・10倍以下・期待値1.2以上」（keirin/market_model.py + scripts/odds_watch.py の ver1 ルール）。
ここにあるのは ver1 を決めるまでに使った検証スクリプトだけ。取得したデータ・結果の数字は入れない（netkeirin 規約 第14条: 私的利用のみ）。
スクリプトは作業用フォルダ（scratchpad）とローカルの data/keirin.sqlite を前提にしたもの。パスは当時のまま。

主な流れ:
- odds_val.py / cross*.py / prereg_check3.py: オッズだけのモデルと1レース1点ルール、事前登録の確認
- sim_all.py / wf_all.py / ai_ev*.py: AI予想の半年シミュレーション、AI確率×オッズ
- yardstick*.py / step3*.py / step4.py / step5*.py: 人気にAIを混ぜたときの上乗せ（物差し）と手順1〜5
- wf_art.py / wf_bp.py / rider_stats.py / select_races.py / comment_test.py: 記事の要素・得意位置・競輪場・選手コメント
- tri_*.py / head_tri.py / mrt_rule.py / banete46.py / leader46.py: 3連単と各種の頭の選び方
