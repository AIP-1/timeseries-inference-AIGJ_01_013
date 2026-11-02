import sys

import numpy as np
import pandas as pd

# 사용법: python baseline.py co2 | temperature | humidity
DIR = "/home/prettygeun/ts-data/"
MEAS = sys.argv[1] if len(sys.argv) > 1 else "co2"
TAG = "" if MEAS == "co2" else "_" + MEAS          # CO₂는 예전 파일 이름 그대로
room = pd.read_pickle(DIR + f"{MEAS}_room_5min.pkl")
splits = {"검증": ("2026-09-21", "2026-10-05"), "시험": ("2026-10-05", "2026-10-19")}
horizons = {"5분": 1, "15분": 3, "30분": 6, "60분": 12}  # 5분 한 칸 기준
rows, points = [], []

for c in room.columns:
    y = room[c]
    for hn, h in horizons.items():
        df = pd.DataFrame({
            "정답": y.shift(-h),             # t+h 시점의 실제 값
            "직전값": y,                      # t 시점 값을 그대로
            "1시간평균": y.rolling(12).mean(),  # t까지 1시간 평균
            "어제같은시각": y.shift(288 - h),   # t+h의 24시간 전 값
        })
        for sn, (s, e) in splits.items():
            part = df[(df.index >= s) & (df.index < e)].dropna()  # 세 모델이 같은 시점에서 평가되도록
            points.append(pd.DataFrame({"공간": c, "구간": sn, "예측거리": hn, "시각": part.index}))
            for m in ["직전값", "1시간평균", "어제같은시각"]:
                err = part[m] - part["정답"]
                rows.append([c, sn, hn, m, len(part), err.abs().mean(), np.sqrt((err ** 2).mean())])

r = pd.DataFrame(rows, columns=["공간", "구간", "예측거리", "모델", "평가점수", "MAE", "RMSE"]).round(2)
r.to_csv(DIR + f"baseline_result{TAG}.csv", index=False)
pd.concat(points).to_csv(DIR + f"eval_points{TAG}.csv", index=False)

pd.set_option("display.width", 200)
order = list(horizons)
for sn in splits:
    print(f"\n[{sn}] {MEAS} MAE")
    t = r[r["구간"] == sn].pivot_table(index=["공간", "모델"], columns="예측거리", values="MAE")[order]
    print(t)
