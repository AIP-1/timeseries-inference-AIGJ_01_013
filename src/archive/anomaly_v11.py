import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 과제 4 — 이상 탐지
# 키트의 "고장 정답 주"가 없어서, 시험 기간(10/5~10/18)에 고장 5종을 직접 심고 그 위치를 정답으로 씀.
# 학습: 학습 기간(정상)만 / 임계값: 검증 기간(정상)으로 결정 / 평가: 고장을 심은 시험 기간
# 먼저 clean.py co2, clean.py temperature를 실행해 둘 것
DIR = "/home/prettygeun/ts-data/"
TRAIN_END, VAL_END, TEST_END = "2026-09-21", "2026-10-05", "2026-10-19"
W = 12                                  # 1시간 창
MERGE = pd.Timedelta("30min")           # 30분 안에 이어지는 경보는 한 건으로 묶음
TOL = pd.Timedelta("1h")                # 고장이 끝난 뒤 1시간 안의 경보(복구 순간의 급변)는 헛경보로 세지 않음
# v2: 탐지는 고장이 "진행 중"일 때 울린 경보만 인정 (v1은 끝난 뒤 1시간까지 인정해서, 복구 순간 급변을 탐지로 잘못 셈)
#     느린 고장용 규칙 추가: 2시간 동안 온도 +1℃ 넘게 상승
# v3: 영점 틀어짐용 "계단" 규칙 + 환기 정지용 "예측 잔차" 탐지(1시간 전에 한 예측보다 실제가 많이 높으면 경보)
# v4: 검증 기간 헛경보 분석 후 규칙 조정 — 급변 500ppm, 값 고정 1시간, 온도 상승은 일하는 시간만
# v5: 검증 기간에도 (시험과 다른 위치에) 고장을 심어, 오토인코더·예측 잔차 임계값을 따로 고름
#     → "검증 헛경보 하루 1회 이하" 안에서 검증 탐지율이 가장 높은 조합 선택 (시험 기간은 여전히 안 봄)
# v6: 고장 모양을 README에 맞춤 — 영점 틀어짐: 6시간에 걸쳐 300~500ppm 서서히 / 냉방 정지: 수업 중 3~5℃ /
#     환기 정지: 사용 중(CO₂가 오르는 때) 상승 폭 1.8배. 탐지 방법은 v5 그대로
# v7: 느린 고장용 탐지기 3개 추가 (임계값은 검증 기간 분위수, 셋이 같은 분위수를 씀)
#     바닥값: 사람 없을 시간(밤·쉬는 날) CO₂가 지난 7일 빈 시간 바닥보다 많이 높음 → 영점 틀어짐
#     3시간 예측 잔차: 3시간 전에 한 예측보다 실제가 많이 높음 → 서서히 쌓이는 영점 틀어짐·환기 정지
#     상승 속도: 1시간 상승 폭이 평소보다 큼 → 환기 정지
# v8: (1) 탐지기마다 임계값을 따로 고름 — 후보(모델 탐지기 × 분위수, 규칙 5개)를 하나씩 더해 보며, 검증 헛경보 한도 안에서
#         검증 탐지율이 가장 많이 오르는 것을 추가 (탐지율이 더 안 오르면 멈춤). 헛경보만 많은 규칙은 빠질 수 있음
#     (2) 검증 기간에 고장을 3벌(90건) 심어 임계값 선택이 우연에 덜 흔들리게 함
#     (3) 탐지기 후보 추가: 30분 예측 잔차, 공간 간 차이(빈 시간에 다른 공간보다 유난히 높음), 온도 2시간 상승(시간 무관)
#     (4) 냉방·환기 정지는 "수업 중"이므로 공휴일에는 심지 않음 (v7까지는 주말만 뺐음 → 시험 고장 위치가 v7과 다름)
# v9: v8은 "탐지 수가 가장 많이 느는 것"부터 넣어서, 헛경보는 적고 영점 틀어짐을 잘 잡는 공간간차이가 한도에 밀려 빠짐
#     → "늘어난 탐지 ÷ 늘어난 헛경보"(효율) 순으로도 골라 보고, 두 방식 중 검증 탐지율이 높은 쪽을 씀
# v10: v9는 시험 탐지 83%였지만 헛경보 1.36회/일 (검증 0.83회보다 많음). 느린 고장은 몇 시간 이어지므로
#      모델 탐지기마다 "15분 이상 계속 넘을 때만 경보" 버전을 후보에 추가 → 잠깐 튀는 헛경보를 거름
# v11: 검증 헛경보가 시험보다 늘 낮게 나온 원인 = 임계값(검증 분위수)을 만든 바로 그 기간에서 헛경보를 셈.
#      → 선택할 때는 검증 앞주(9/21~27)로 만든 임계값을 뒷주(9/28~10/4)에, 뒷주 임계값을 앞주에 적용해 헛경보를 셈
#        (처음 보는 기간에 쓸 때와 같은 조건). 최종 임계값은 이전처럼 검증 2주 전체로 만듦
rng = np.random.default_rng(0)
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

co2 = pd.read_pickle(DIR + "co2_room_5min.pkl")
tmp = pd.read_pickle(DIR + "temperature_room_5min.pkl")
rooms = list(co2.columns)
idx = co2.index
test_days = (pd.Timestamp(TEST_END) - pd.Timestamp(VAL_END)).days
offday = ((idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin([
    "2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05", "2026-10-09"])).astype(float)


# ---------- 1. 시험 기간에 고장 심기 ----------
# (종류, 센서, 길이(5분 칸), 낮 시간에만?)
FAULTS = [("튄 값", "co2", 1, False), ("값 고정", "co2", 36, False), ("영점 틀어짐", "co2", 72, False),
          ("냉방 정지", "temp", 48, True), ("환기 정지", "co2", 36, True)]
N_EACH = 2                              # 공간마다 종류별 2번 → 3공간 × 5종 × 2 = 30건


def inject(c, t, r, start, end):
    """공간 r의 CO₂(c)·온도(t)의 [start, end) 구간에 고장을 심고 정답 목록을 돌려줌"""
    c, t = c.copy(), t.copy()
    test_pos = np.where((idx >= start) & (idx < end))[0]
    busy = np.zeros(len(idx), bool)
    labels = []
    for kind, sensor, L, daytime in FAULTS:
        for _ in range(N_EACH):
            for _try in range(1000):
                s = rng.choice(test_pos[:-L - 12])
                ts = idx[s]
                if daytime and (offday[s] or not 9 <= ts.hour <= 14):
                    continue
                if busy[max(0, s - 36):s + L + 36].any() or np.isnan(c[s:s + L]).any() or np.isnan(t[s:s + L]).any():
                    continue
                if kind == "환기 정지" and c[s:s + L].max() - c[s] < 100:   # 사람이 있어 CO₂가 100ppm 넘게 오르는 구간만
                    continue
                break
            busy[s:s + L] = True
            if kind == "튄 값":
                c[s] += 800
            elif kind == "값 고정":
                c[s:s + L] = c[s]
            elif kind == "영점 틀어짐":
                c[s:s + L] += np.linspace(0, rng.uniform(300, 500), L)   # 6시간에 걸쳐 300~500ppm 서서히 틀어짐
            elif kind == "냉방 정지":
                t[s:s + L] += np.linspace(0, rng.uniform(3, 5), L)       # 수업 중 4시간 동안 3~5℃ 상승
            elif kind == "환기 정지":
                d = np.diff(c[s:s + L], prepend=c[s])
                c[s:s + L] = c[s] + np.cumsum(np.where(d > 0, 1.8 * d, d))   # 오를 때마다 1.8배로 오름 (3시간)
            labels.append({"공간": r, "종류": kind, "센서": sensor, "시작": idx[s], "끝": idx[s + L - 1]})
    return c, t, labels


data, labels = {}, []
for r in rooms:
    c, t, lab = inject(co2[r].values, tmp[r].values, r, VAL_END, TEST_END)
    data[r] = (c, t)
    labels += lab
labels = pd.DataFrame(labels)
labels.to_csv(DIR + "anomaly_labels.csv", index=False)

# 임계값 고르기용: 검증 기간에도 고장을 3벌 심음 (시험과 독립, 벌마다 다른 위치)
N_REP = 3
val_reps = []
for _ in range(N_REP):
    d, labs = {}, []
    for r in rooms:
        c, t, lab = inject(co2[r].values, tmp[r].values, r, TRAIN_END, VAL_END)
        d[r] = (c, t)
        labs += lab
    val_reps.append((d, pd.DataFrame(labs)))

# ---------- 2. 규칙 탐지 ----------
def rule_alarm(c, t, parts=False):
    dc, dt = np.abs(np.diff(c, prepend=np.nan)), np.abs(np.diff(t, prepend=np.nan))
    jump = (dc > 500) | (dt > 2)                                  # 급변 (v4: 300→500, 회의 시작 급상승 헛경보 줄임)
    stuck = pd.Series(c).rolling(12).std().values < 1e-6           # 1시간 동안 값이 똑같음 (v4: 30분→1시간, 10분 간격 센서 고려)
    out = (c < 350) | (c > 5000) | (t < 0) | (t > 45)              # 범위 밖
    work = ~offday.astype(bool) & (idx.hour >= 9) & (idx.hour < 18)
    warm = ((t - np.roll(t, 24)) > 1.0) & work                     # 일하는 시간(냉방이 돌아야 할 때) 2시간 +1℃ 넘게 상승
    warm[:24] = False                                              # (v4: 퇴근 후 냉방 꺼져서 오르는 건 정상이라 제외)
    # 계단: 직전 30분과 최근 30분이 둘 다 잔잔한데 평균이 120ppm 넘게 차이 (영점 틀어짐)
    cs = pd.Series(c)
    prev_m, prev_s = cs.shift(6).rolling(6).mean(), cs.shift(6).rolling(6).std()
    next_m, next_s = cs.rolling(6).mean(), cs.rolling(6).std()
    step = ((next_m - prev_m).abs() > 120) & (prev_s < 25) & (next_s < 25)
    step = step.values
    if parts:
        return {"급변": jump, "값 고정": stuck, "범위 밖": out, "온도 상승": warm, "계단": step}
    cause = np.where((dt > 2) | warm, "temp", "co2")
    return jump | stuck | out | warm | step, cause


# ---------- 3. 오토인코더 (정상 구간만 학습) ----------
tr_mask = idx < TRAIN_END
mc, sc = np.nanmean(co2[tr_mask].values), np.nanstd(co2[tr_mask].values)
mt, st = np.nanmean(tmp[rooms][tr_mask].values), np.nanstd(tmp[rooms][tr_mask].values)


def windows(c, t):
    """창 끝 시각 기준 [CO₂ 수준, 온도 수준, CO₂ 변화, 온도 변화] × 12칸"""
    f = np.column_stack([(c - mc) / sc, (t - mt) / st,
                         np.diff(c, prepend=np.nan) / 50, np.diff(t, prepend=np.nan) / 0.2])
    X = np.full((len(c), W, 4), np.nan, np.float32)
    for i in range(W - 1, len(c)):
        X[i] = f[i - W + 1:i + 1]
    ok = ~np.isnan(X).any(axis=(1, 2))
    return X, ok


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(W * 4, 32), nn.ReLU(), nn.Linear(32, 8))
        self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, W * 4))

    def forward(self, x):
        return self.dec(self.enc(x)).view(-1, W, 4)


Xtr = np.concatenate([w[tr_mask & ok] for w, ok in [windows(co2[r].values, tmp[r].values) for r in rooms]])
model = AE().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xt = torch.from_numpy(Xtr).to(dev)
for ep in range(30):
    perm = torch.randperm(len(Xt), device=dev)
    for i in range(0, len(Xt), 256):
        b = Xt[perm[i:i + 256]]
        opt.zero_grad()
        loss = ((model(b) - b) ** 2).mean()
        loss.backward()
        opt.step()
print(f"오토인코더 학습 끝 (학습 창 {len(Xtr)}개, 마지막 손실 {loss.item():.4f})")
torch.save(model.state_dict(), DIR + "ae_anomaly.pt")


def ae_score(c, t):
    """창 끝 시각별 복원 오차와, CO₂ 쪽/온도 쪽 중 어느 쪽 오차가 더 큰지"""
    X, ok = windows(c, t)
    score = np.full(len(c), np.nan)
    cause = np.full(len(c), "co2", dtype=object)
    model.eval()
    with torch.no_grad():
        Xo = torch.from_numpy(np.nan_to_num(X[ok])).to(dev)
        err = ((model(Xo) - Xo) ** 2).cpu().numpy()
    score[ok] = err.mean(axis=(1, 2))
    cause[ok] = np.where(err[:, :, [1, 3]].sum(axis=(1, 2)) > err[:, :, [0, 2]].sum(axis=(1, 2)), "temp", "co2")
    return score, cause


# ---------- 3-2. 예측 잔차 (환기 정지처럼 천천히 더 오르는 고장용) ----------
# 지난 2시간 CO₂ 모양 + 시각으로 1시간 뒤 변화량을 예측하는 선형 모델(정상 학습 기간만으로 학습).
# 실제 값이 1시간 전에 한 예측보다 많이 높으면 이상.
H = 12
hour = idx.hour + idx.minute / 60


def lin_feats(c, h=H):
    """t 시점 특징: 지난 2시간의 '지금 대비 차이' 24개 + h칸 뒤 시각(sin, cos) + 쉬는 날 + 상수"""
    n = len(c)
    F = np.full((n, 24 + 4), np.nan)
    lag = np.column_stack([np.roll(c, k) for k in range(24)])  # lag[:,k] = c[t-k]
    lag[:24] = np.nan
    F[:, :24] = (lag - c[:, None]) / 50
    th = np.roll(hour, -h)
    F[:, 24], F[:, 25] = np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24)
    F[:, 26], F[:, 27] = np.roll(offday, -h), 1.0
    return F


def fit_ridge(h):
    Fs, Ys = [], []
    for r in rooms:
        c = co2[r].values
        F, y = lin_feats(c, h), (np.roll(c, -h) - c) / 50
        m = tr_mask & ~np.isnan(F).any(axis=1) & ~np.isnan(y)
        m[-h:] = False
        Fs.append(F[m]), Ys.append(y[m])
    Fa, Ya = np.concatenate(Fs), np.concatenate(Ys)
    return np.linalg.solve(Fa.T @ Fa + 1.0 * np.eye(Fa.shape[1]), Fa.T @ Ya)   # 릿지 회귀


H3 = 36                                 # 3시간
beta, beta3 = fit_ridge(H), fit_ridge(H3)


def resid_score(c, h=H):
    """실제 CO₂ - (h칸 전 값 + h칸 전에 한 예측), 15분 평균"""
    pred_change = lin_feats(c, h) @ (beta if h == H else beta3) * 50
    expected = np.roll(c + pred_change, h)
    expected[:h] = np.nan
    return pd.Series(c - expected).rolling(3).mean().values


# ---------- 3-3. 느린 고장용 탐지기 (v7) ----------
empty = (offday == 1) | (idx.hour < 7) | (idx.hour >= 21)      # 사람이 없을 시간 (가정)


def floor_score(c):
    """빈 시간의 최근 1시간 최솟값 - 하루 전까지 7일간 빈 시간 하위 10% (빈 시간이 아니면 NaN)"""
    s = pd.Series(np.where(empty, c, np.nan), index=idx)
    ref = s.rolling("7D", min_periods=36).quantile(0.1).shift(288)   # 하루 밀어서 지금 고장이 기준에 섞이지 않게
    cur = s.rolling(12, min_periods=12).min()
    return (cur - ref).values


def rise_score(c):
    """1시간 상승 폭"""
    return c - np.roll(c, 12)


def slow_scores(c):
    return {"바닥값": floor_score(c), "3시간잔차": resid_score(c, H3), "상승속도": rise_score(c)}


# ---------- 4. 경보 묶기와 평가 ----------
def events(alarm, cause, r, start, end):
    """연속 경보를 한 건으로 묶음 (30분 안에 다시 울리면 같은 건)"""
    m = alarm & (idx >= start) & (idx < end)
    ev = []
    for i in np.where(m)[0]:
        if ev and idx[i] - ev[-1]["끝"] <= MERGE:
            ev[-1]["끝"] = idx[i]
        else:
            ev.append({"공간": r, "시작": idx[i], "끝": idx[i], "원인": cause[i]})
    return ev


def evaluate(evs, labs, days=None):
    """종류별 탐지율·평균 지연(분), 하루당 헛경보(공간 합)"""
    evs = pd.DataFrame(evs, columns=["공간", "시작", "끝", "원인"])
    rows, used = [], np.zeros(len(evs), bool)
    for _, l in labs.iterrows():
        hit = (evs["공간"] == l["공간"]) & (evs["끝"] >= l["시작"]) & (evs["시작"] <= l["끝"])
        used |= ((evs["공간"] == l["공간"]) & (evs["끝"] >= l["시작"]) & (evs["시작"] <= l["끝"] + TOL)).values
        if hit.any():
            first = evs[hit]["시작"].min()
            rows.append([l["종류"], 1, max(0, (first - l["시작"]).total_seconds() / 60),
                         (evs[hit]["원인"] == l["센서"]).any()])
        else:
            rows.append([l["종류"], 0, np.nan, False])
    per = pd.DataFrame(rows, columns=["종류", "탐지", "지연분", "원인맞음"])
    fa = (~used).sum() / (days or test_days)
    return per, fa


def val_fa(alarm_fn):
    """검증 기간(고장 없음)의 하루당 헛경보, 공간 합"""
    n = 0
    for r in rooms:
        a, cause = alarm_fn(r, co2[r].values, tmp[r].values)
        n += len(events(a, cause, r, TRAIN_END, VAL_END))
    return n / (pd.Timestamp(VAL_END) - pd.Timestamp(TRAIN_END)).days


# ---------- 5. 탐지기 후보와 임계값 선택 (v8) ----------
def pool(r, D):
    """공간 r의 탐지기 후보 점수 {이름: (점수, 원인 센서)}. D = {공간: (CO₂, 온도)} (공간 간 비교에 다른 공간도 씀)"""
    c, t = D[r]
    ae, ae_cause = ae_score(c, t)
    others = np.nanmean(np.column_stack([D[o][0] for o in rooms if o != r]), axis=1)
    diff = pd.Series(np.where(empty, c - others, np.nan), index=idx)
    diff_ref = diff.rolling("7D", min_periods=36).median().shift(288)
    return {
        "오토인코더": (ae, ae_cause),
        "1시간잔차": (resid_score(c), "co2"),
        "30분잔차": (pd.Series(c - np.roll(c + lin_feats(c, 6) @ beta6 * 50, 6)).rolling(3).mean().values, "co2"),
        "3시간잔차": (resid_score(c, H3), "co2"),
        "바닥값": (floor_score(c), "co2"),
        "공간간차이": (diff.rolling(12, min_periods=12).min().values - diff_ref.values, "co2"),
        "상승속도": (rise_score(c), "co2"),
        "온도2시간상승": (t - np.roll(t, 24), "temp"),
    }


beta6 = fit_ridge(6)
BASE = ["오토인코더", "1시간잔차", "30분잔차", "3시간잔차", "바닥값", "공간간차이", "상승속도", "온도2시간상승"]
NAMES = BASE + [n + "·15분지속" for n in BASE]
RULES = ["급변", "값 고정", "범위 밖", "온도 상승", "계단"]
QS = [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]
vmask = (idx >= TRAIN_END) & (idx < VAL_END)
vdays = (pd.Timestamp(VAL_END) - pd.Timestamp(TRAIN_END)).days

# 임계값 = 고장 없는 원래 검증 기간 점수의 공간별 분위수
clean = {r: (co2[r].values, tmp[r].values) for r in rooms}
clean_pool = {r: pool(r, clean) for r in rooms}
THR = {(r, n, q): np.nanquantile(clean_pool[r][n][0][vmask], q) for r in rooms for n in BASE for q in QS}
MID = "2026-09-28"
wk1, wk2 = (idx >= TRAIN_END) & (idx < MID), (idx >= MID) & (idx < VAL_END)
# 선택용: 칸마다 "다른 주"로 만든 임계값
THR_X = {(r, n, q): np.where(wk1, np.nanquantile(clean_pool[r][n][0][wk2], q), np.nanquantile(clean_pool[r][n][0][wk1], q))
         for r in rooms for n in BASE for q in QS}


def base(n):
    return n.split("·")[0]


def alarms(P, r, n, q, cross=False):
    b = base(n)
    sc, _ = P[r][b]
    a = np.nan_to_num(sc, nan=-1e9) > (THR_X if cross else THR)[(r, b, q)]
    if b == "온도2시간상승":
        a[:24] = False
    if n.endswith("15분지속"):                     # 지금까지 3칸(15분) 연속으로 넘었을 때만
        a = a & np.roll(a, 1) & np.roll(a, 2)
    return a


def quick(alarm_by_room, labs, start, end, days):
    """빠른 채점: (탐지율, 하루당 헛경보). 묶기·인정 규칙은 evaluate와 같음"""
    det, fa_n = [], 0
    m = (idx >= start) & (idx < end)
    for r in rooms:
        pos = np.where(alarm_by_room[r] & m)[0]
        if len(pos):
            br = np.where(np.diff(pos) > 6)[0]                    # 30분(6칸) 넘게 떨어지면 다른 건
            ev = np.column_stack([pos[np.r_[0, br + 1]], pos[np.r_[br, len(pos) - 1]]])
        else:
            ev = np.zeros((0, 2), int)
        used = np.zeros(len(ev), bool)
        for _, l in labs[labs["공간"] == r].iterrows():
            s0, e0 = idx.get_loc(l["시작"]), idx.get_loc(l["끝"])
            det.append(((ev[:, 1] >= s0) & (ev[:, 0] <= e0)).any())
            used |= (ev[:, 1] >= s0) & (ev[:, 0] <= e0 + 12)
        fa_n += (~used).sum()
    return float(np.mean(det)), fa_n / days


# 검증 3벌의 점수·경보를 미리 계산
rep_pool = [{r: pool(r, d) for r in rooms} for d, _ in val_reps]
rep_rule = [{r: rule_alarm(*d[r])[0] for r in rooms} for d, _ in val_reps]
rep_alarm = {(k, n, q): {r: alarms(rep_pool[k], r, n, q, cross=True) for r in rooms}
             for k in range(N_REP) for n in NAMES for q in QS}
for k, (d, _) in enumerate(val_reps):                     # 규칙은 분위수 없이 하나씩 (q = None)
    parts = {r: rule_alarm(*d[r], parts=True) for r in rooms}
    for n in RULES:
        rep_alarm[(k, n, None)] = {r: parts[r][n].astype(bool) for r in rooms}


def val_score(sel):
    """선택한 (탐지기, 분위수) 목록의 검증 3벌 평균 (탐지율, 헛경보/일)"""
    ds, fs = [], []
    for k, (_, labs) in enumerate(val_reps):
        al = {r: np.zeros(len(idx), bool) for r in rooms}
        for n, q in sel:
            for r in rooms:
                al[r] |= rep_alarm[(k, n, q)][r]
        d, f = quick(al, labs, TRAIN_END, VAL_END, vdays)
        ds.append(d), fs.append(f)
    return np.mean(ds), np.mean(fs)


def select(budget):
    """두 방식(탐지 증가 순 / 효율 순)으로 골라 검증 탐지율이 높은 쪽 (같으면 헛경보 적은 쪽)"""
    a, b = greedy(budget, "gain"), greedy(budget, "ratio")
    return max([a, b], key=lambda x: (round(x[1], 6), -x[2]))


def greedy(budget, mode):
    """빈 상태에서 출발해, 검증 헛경보 ≤ budget 안에서 후보를 하나씩 추가
    mode="gain": 검증 탐지율을 가장 많이 올리는 것 / "ratio": 늘어난 탐지율 ÷ (늘어난 헛경보 + 0.05)가 가장 큰 것"""
    sel = []
    best_d, best_f = 0.0, 0.0
    while True:
        cand = []                                          # (고르는 기준, -헛경보, 이름, 분위수, 탐지율)
        for n in NAMES + RULES:
            if any(base(n) == base(s_[0]) for s_ in sel):     # 같은 탐지기는 한 번만 (지속 버전 포함)
                continue
            for q in (QS if n in NAMES else [None]):
                d, f = val_score(sel + [(n, q)])
                if f <= budget and d > best_d + 1e-9:
                    key = d if mode == "gain" else (d - best_d) / (max(f - best_f, 0) + 0.05)
                    cand.append((key, -f, n, q, d))
        if not cand:
            return sel, best_d, best_f
        _, nf, n, q, d = max(cand, key=lambda x: (x[0], x[1]))
        sel.append((n, q))
        best_d, best_f = d, -nf


# 진단: 탐지기 하나씩(분위수 0.999)일 때 검증 고장 종류별 탐지 수(3벌 합)와 헛경보
diag = []
for n in BASE + RULES + ["규칙(전체)"]:
    row = {"탐지기": n}
    for k, (_, labs) in enumerate(val_reps):
        al = rep_rule[k] if n == "규칙(전체)" else rep_alarm[(k, n, 0.999 if n in NAMES else None)]
        for kind in labs["종류"].unique():
            row[kind] = row.get(kind, 0) + round(quick(al, labs[labs["종류"] == kind], TRAIN_END, VAL_END, vdays)[0] * 6)
        row["헛경보/일"] = row.get("헛경보/일", 0) + quick(al, labs, TRAIN_END, VAL_END, vdays)[1] / N_REP
    diag.append(row)
print(f"\n[검증 진단] 탐지기 하나만 (분위수 0.999), 고장 종류별 탐지 수 (3벌 합, 종류당 {6 * N_REP}건)")
print(pd.DataFrame(diag).set_index("탐지기").round(2).to_string())


P_TEST = {r: pool(r, data) for r in rooms}


def test_fn(sel):
    """선택한 탐지기 목록으로 시험 기간 경보와 원인 센서"""
    def f(r, c, t):
        a = np.zeros(len(c), bool)
        cause = np.full(len(c), "co2", dtype=object)
        parts, rcause = rule_alarm(c, t, parts=True), rule_alarm(c, t)[1]
        for n, q in sel:
            if n in RULES:
                an, cs = parts[n].astype(bool), rcause
            else:
                an, cs = alarms(P_TEST, r, n, q), P_TEST[r][base(n)][1]
            cause = np.where(an & ~a, cs, cause)
            a |= an
        return a, cause
    return f


def run_test(fn):
    evs = []
    for r in rooms:
        c, t = data[r]
        a, cause = fn(r, c, t)
        evs += events(a, cause, r, VAL_END, TEST_END)
    return evs


BUDGET = 1.0
sel, vd, vf = select(BUDGET)
print(f"\n[선택된 탐지기 (검증 헛경보 {BUDGET}회/일 이하)] {sel}  → 검증 탐지율 {vd * 100:.0f}%, 검증 헛경보 {vf:.2f}회/일")

summary = []
for name, fn in [("모델", test_fn([x for x in sel if x[0] in NAMES])), ("규칙(전체)", test_fn([(n, None) for n in RULES])),
                 ("결합", test_fn(sel))]:
    evs = run_test(fn)
    per, fa = evaluate(evs, labels)
    g = per.groupby("종류").agg(탐지=("탐지", "sum"), 건수=("탐지", "size"), 지연분=("지연분", "mean"))
    print(f"\n[시험·{name}] 탐지율 {per['탐지'].mean() * 100:.0f}%  헛경보 {fa:.2f}회/일"
          f"  원인 센서 맞음 {per.loc[per['탐지'] == 1, '원인맞음'].mean() * 100:.0f}%")
    print(g.round(0).to_string())
    summary.append([name, round(per["탐지"].mean() * 100), round(fa, 2)])
    if name == "결합":
        pd.DataFrame(evs).to_csv(DIR + "anomaly_alarms.csv", index=False)

print("\n[성적표]\n", pd.DataFrame(summary, columns=["방식", "탐지율%", "헛경보/일"]).to_string(index=False))

# 헛경보 한도를 바꿔 가며 (선택은 매번 검증으로, 시험은 결과 보고용)
sweep = []
for b in [0.5, 0.75, 1.0, 1.5, 2.0]:
    sb, bd, bf = select(b)
    per, fa = evaluate(run_test(test_fn(sb)), labels)
    sweep.append([b, round(bd * 100), round(bf, 2), round(per["탐지"].mean() * 100), round(fa, 2), len(sb)])
print("\n[헛경보 한도별 결합 성적]\n", pd.DataFrame(sweep, columns=["검증한도", "검증탐지율%", "검증헛경보/일",
                                                          "시험탐지율%", "시험헛경보/일", "탐지기수"]).to_string(index=False))
