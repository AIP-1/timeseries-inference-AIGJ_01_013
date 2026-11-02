import numpy as np
import pandas as pd

co2 = pd.read_pickle("/home/prettygeun/ts-data/co2_12w.pkl")
d = co2[["time", "device_id", "location", "value"]].dropna().sort_values(["device_id", "time"])
n0 = len(d)

# 규칙 1. 불가능한 값
d = d[d["value"].between(350, 5000)]
n1 = len(d)

# 규칙 3. 중복: 같은 센서, 5초 이내, 같은 값
dt = d.groupby("device_id")["time"].diff()
dv = d.groupby("device_id")["value"].diff()
d = d[~((dt <= pd.Timedelta("5s")) & (dv == 0))]
n3 = len(d)

# 규칙 2. 튄 값: 앞뒤 값 둘 다와 300ppm 넘게, 같은 방향으로 차이
prev = d.groupby("device_id")["value"].shift(1)
nxt = d.groupby("device_id")["value"].shift(-1)
a, b = d["value"] - prev, d["value"] - nxt
spike = (a.abs() > 300) & (b.abs() > 300) & (np.sign(a) == np.sign(b))
d = d[~spike]
print(f"원본 {n0} / 범위밖 {n0-n1} / 중복 {n1-n3} / 튄값 {spike.sum()} / 남음 {len(d)}")

# 규칙 4. 센서별 5분 평균
d["slot"] = d["time"].dt.floor("5min")
dev = d.groupby(["location", "device_id", "slot"])["value"].mean().reset_index()

# 10분 간격 센서는 5분 칸이 하나 걸러 비므로, 센서별로 앞 값 1칸 채운 뒤 합침
idx = pd.date_range("2026-07-27", "2026-10-19", freq="5min", tz="Asia/Seoul", inclusive="left")
wide = dev.pivot_table(index="slot", columns=["location", "device_id"], values="value")
wide = wide.reindex(idx).ffill(limit=1)

# 규칙 6·7. 공간별 합치기: 평소엔 전체 평균, 하나가 150ppm 넘게 튀면 중앙 두 값 평균
def combine(v):
    v = v.dropna().sort_values()
    if len(v) == 0:
        return np.nan
    if len(v) >= 3 and (v - v.median()).abs().max() > 150:
        k = len(v) // 2
        return v.iloc[k - 1:k + 1].mean() if len(v) % 2 == 0 else v.median()
    return v.mean()

room = pd.DataFrame({loc: wide[loc].apply(combine, axis=1) for loc in wide.columns.get_level_values(0).unique()})

# 규칙 5. 15분(3칸) 이하로 비어 있는 구간만 직선으로 채움
def fill_short(s, n=3):
    na = s.isna()
    run = na.groupby((~na).cumsum()).transform("sum")
    return s.where(~na | (run > n), s.interpolate(limit_area="inside"))

for c in room.columns:
    room[c] = fill_short(room[c])

room.to_pickle("/home/prettygeun/ts-data/co2_room_5min.pkl")
print(room.shape)
print("결측 비율:\n", room.isna().mean().round(3))
print(room.describe().round(0))
