import sys

import pandas as pd

from features import CFG, clean_room

# 사용법: python clean.py co2 | temperature | humidity
# v2: 정제 규칙을 features.py(clean_room)로 옮김 — 상시 서비스(service.py)와 같은 코드를 씀. 결과는 v1과 같음
DIR = "/home/prettygeun/ts-data/"
MEAS = sys.argv[1] if len(sys.argv) > 1 else "co2"

raw = pd.read_pickle(DIR + CFG[MEAS][0])
idx = pd.date_range("2026-07-27", "2026-10-19", freq="5min", tz="Asia/Seoul", inclusive="left")
room = clean_room(raw, MEAS, idx, verbose=True)

room.to_pickle(DIR + f"{MEAS}_room_5min.pkl")
print("공간별 센서 수:", raw[raw["location"] != "lab"].groupby("location")["device_id"].nunique().to_dict())
print(room.shape)
print("결측 비율:\n", room.isna().mean().round(3))
print(room.describe().round(0))
