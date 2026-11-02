import numpy as np
import pandas as pd

# 학습 스크립트와 상시 서비스(service.py)가 함께 쓰는 전처리·특징 함수 (과제 8: 복사하지 않고 import해서 같은 코드 사용)
#   clean_room     : 원본 → 공간별 5분 값 (clean.py 규칙 1~7)
#   gru_input      : 과제 3 GRU 입력 (지금 값 대비 차이 + 시각 + 공간)
#   ae_windows     : 과제 4 오토인코더 입력 (1시간 창)
#   rule_alarm     : 과제 4 규칙 탐지
#   occ_features   : 과제 5 사용 여부 MLP 입력
HOLIDAYS = ["2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05", "2026-10-09"]
# 항목별: 원본 파일, 가능한 범위, 튄 값 기준, 센서 합칠 때 "튄다" 기준
CFG = {
    "co2":         ("co2_12w.pkl",  350, 5000, 300, 150),
    "temperature": ("temp_12w.pkl",   0,   45,   3, 1.5),
    "humidity":    ("hum_12w.pkl",    0,  100,  10,   5),
}


def offday(idx):
    return (idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin(HOLIDAYS)


def clean_room(raw, meas, idx, verbose=False):
    """원본(time, device_id, location, value) → 공간별 5분 값 DataFrame (index = idx)"""
    _, LO, HI, SPIKE, OUT = CFG[meas]
    raw = raw[raw["location"] != "lab"]       # dev001/lab은 테스트용 데이터
    d = raw[["time", "device_id", "location", "value"]].dropna().sort_values(["device_id", "time"])
    n0 = len(d)

    # 규칙 1. 불가능한 값
    d = d[d["value"].between(LO, HI)]
    n1 = len(d)

    # 규칙 3. 중복: 같은 센서, 5초 이내, 같은 값
    dt = d.groupby("device_id")["time"].diff()
    dv = d.groupby("device_id")["value"].diff()
    d = d[~((dt <= pd.Timedelta("5s")) & (dv == 0))]
    n3 = len(d)

    # 규칙 2. 튄 값: 앞뒤 값 둘 다와 SPIKE 넘게, 같은 방향으로 차이
    prev = d.groupby("device_id")["value"].shift(1)
    nxt = d.groupby("device_id")["value"].shift(-1)
    a, b = d["value"] - prev, d["value"] - nxt
    spike = (a.abs() > SPIKE) & (b.abs() > SPIKE) & (np.sign(a) == np.sign(b))
    d = d[~spike]
    if verbose:
        print(f"원본 {n0} / 범위밖 {n0-n1} / 중복 {n1-n3} / 튄값 {spike.sum()} / 남음 {len(d)}")

    # 규칙 4. 센서별 5분 평균
    d = d.assign(slot=d["time"].dt.floor("5min"))
    dev = d.groupby(["location", "device_id", "slot"])["value"].mean().reset_index()

    # 10분 간격 센서는 5분 칸이 하나 걸러 비므로, 센서별로 앞 값 1칸 채운 뒤 합침
    wide = dev.pivot_table(index="slot", columns=["location", "device_id"], values="value")
    wide = wide.reindex(idx).ffill(limit=1)

    # 규칙 6·7. 공간별 합치기: 평소엔 전체 평균, 하나가 OUT 넘게 튀면 중앙 두 값 평균
    def combine(v):
        v = v.dropna().sort_values()
        if len(v) == 0:
            return np.nan
        if len(v) >= 3 and (v - v.median()).abs().max() > OUT:
            k = len(v) // 2
            return v.iloc[k - 1:k + 1].mean() if len(v) % 2 == 0 else v.median()
        return v.mean()

    room = pd.DataFrame({loc: wide[loc].apply(combine, axis=1) for loc in wide.columns.get_level_values(0).unique()},
                        index=idx)

    # 규칙 5. 15분(3칸) 이하로 비어 있는 구간만 직선으로 채움
    def fill_short(s, n=3):
        na = s.isna()
        run = na.groupby((~na).cumsum()).transform("sum")
        return s.where(~na | (run > n), s.interpolate(limit_area="inside"))

    for c in room.columns:
        room[c] = fill_short(room[c])
    return room


def time_feat(idx):
    hour = idx.hour + idx.minute / 60
    return np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), offday(idx).astype(float)], axis=1)


def gru_input(c, t, idx, room_i, cfg, n_in):
    """공간 CO₂ c(ppm 배열)의 t 시점 GRU 입력 (LB, n_in). 창에 결측이 있으면 None"""
    LB, mu, sd, DS, n_rooms = cfg["lookback"], cfg["mu"], cfg["sd"], cfg["dscale"], len(cfg["rooms"])
    if t < LB - 1:
        return None
    win = (c[t - LB + 1:t + 1] - mu) / sd
    if np.isnan(win).any():
        return None
    cols = [(win - win[-1]) * sd / DS, time_feat(idx[t - LB + 1:t + 1]), np.tile(np.eye(n_rooms)[room_i], (LB, 1))]
    if n_in > 1 + 3 + n_rooms:                           # 일정 칸이 있는 버전이면 0(일정 없음)으로 채움
        cols.append(np.zeros((LB, n_in - 1 - 3 - n_rooms)))
    return np.column_stack(cols).astype(np.float32)


def ae_features(c, t, mc, sc, mt, st):
    """칸별 [CO₂ 수준, 온도 수준, CO₂ 변화, 온도 변화] (정규화)"""
    return np.column_stack([(c - mc) / sc, (t - mt) / st, np.diff(c, prepend=np.nan) / 50, np.diff(t, prepend=np.nan) / 0.2])


def ae_window(f, i, W=12):
    w = f[i - W + 1:i + 1]
    return None if i < W - 1 or np.isnan(w).any() else w.astype(np.float32)


def rule_alarm(c, t, idx, parts=False):
    """과제 4 규칙 탐지 (anomaly.py와 같은 기준)"""
    dc, dt = np.abs(np.diff(c, prepend=np.nan)), np.abs(np.diff(t, prepend=np.nan))
    jump = (dc > 500) | (dt > 2)                                  # 급변
    stuck = pd.Series(c).rolling(12).std().values < 1e-6           # 1시간 동안 값이 똑같음
    out = (c < 350) | (c > 5000) | (t < 0) | (t > 45)              # 범위 밖
    work = ~offday(idx) & (idx.hour >= 9) & (idx.hour < 18)
    warm = ((t - np.roll(t, 24)) > 1.0) & work                     # 일하는 시간 2시간 +1℃ 넘게 상승
    warm[:24] = False
    cs = pd.Series(c)
    prev_m, prev_s = cs.shift(6).rolling(6).mean(), cs.shift(6).rolling(6).std()
    next_m, next_s = cs.rolling(6).mean(), cs.rolling(6).std()
    step = (((next_m - prev_m).abs() > 120) & (prev_s < 25) & (next_s < 25)).values   # 계단 (영점 틀어짐)
    if parts:
        return {"급변": jump, "값 고정": stuck, "범위 밖": out, "온도 상승": warm, "계단": step}
    cause = np.where((dt > 2) | warm, "temp", "co2")
    return jump | stuck | out | warm | step, cause


def lag(c, k):
    out = np.roll(c, k)
    out[:k] = np.nan
    return out


def occ_features(c, t, room_i, n_rooms, mc, sc):
    """과제 5 사용 여부 입력: 지난 1시간 CO₂ 변화, 24시간 최저 대비 높이, 온도 변화, 공간"""
    base24 = pd.Series(c).rolling(288, min_periods=36).min().values
    onehot = np.tile(np.eye(n_rooms)[room_i], (len(c), 1))
    return np.column_stack([(c - lag(c, k)) / 50 for k in (1, 2, 3, 6, 12)] +
                           [(c - base24) / 100, (c - mc) / sc, t - lag(t, 3), t - lag(t, 12), onehot])
