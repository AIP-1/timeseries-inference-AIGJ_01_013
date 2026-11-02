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
rng = np.random.default_rng(0)
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

co2 = pd.read_pickle(DIR + "co2_room_5min.pkl")
tmp = pd.read_pickle(DIR + "temperature_room_5min.pkl")
rooms = list(co2.columns)
idx = co2.index
test_days = (pd.Timestamp(TEST_END) - pd.Timestamp(VAL_END)).days


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
                if daytime and (ts.dayofweek >= 5 or not 9 <= ts.hour <= 14):
                    continue
                if busy[max(0, s - 36):s + L + 36].any() or np.isnan(c[s:s + L]).any() or np.isnan(t[s:s + L]).any():
                    continue
                break
            busy[s:s + L] = True
            if kind == "튄 값":
                c[s] += 800
            elif kind == "값 고정":
                c[s:s + L] = c[s]
            elif kind == "영점 틀어짐":
                c[s:s + L] += 200
            elif kind == "냉방 정지":
                t[s:s + L] += np.linspace(0, 2.5, L)        # 4시간 동안 2.5℃ 상승
            elif kind == "환기 정지":
                c[s:s + L] += np.linspace(0, 400, L)        # 3시간 동안 400ppm 추가 상승
            labels.append({"공간": r, "종류": kind, "센서": sensor, "시작": idx[s], "끝": idx[s + L - 1]})
    return c, t, labels


data, labels = {}, []
for r in rooms:
    c, t, lab = inject(co2[r].values, tmp[r].values, r, VAL_END, TEST_END)
    data[r] = (c, t)
    labels += lab
labels = pd.DataFrame(labels)
labels.to_csv(DIR + "anomaly_labels.csv", index=False)

# 임계값 고르기용: 검증 기간에도 고장을 심음 (시험과 독립)
data_val, labels_val = {}, []
for r in rooms:
    c, t, lab = inject(co2[r].values, tmp[r].values, r, TRAIN_END, VAL_END)
    data_val[r] = (c, t)
    labels_val += lab
labels_val = pd.DataFrame(labels_val)

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
offday = ((idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin([
    "2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05", "2026-10-09"])).astype(float)


def lin_feats(c):
    """t 시점 특징: 지난 2시간의 '지금 대비 차이' 24개 + 1시간 뒤 시각(sin, cos) + 쉬는 날 + 상수"""
    n = len(c)
    F = np.full((n, 24 + 4), np.nan)
    lag = np.column_stack([np.roll(c, k) for k in range(24)])  # lag[:,k] = c[t-k]
    lag[:24] = np.nan
    F[:, :24] = (lag - c[:, None]) / 50
    th = np.roll(hour, -H)
    F[:, 24], F[:, 25] = np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24)
    F[:, 26], F[:, 27] = np.roll(offday, -H), 1.0
    return F


Fs, Ys = [], []
for r in rooms:
    c = co2[r].values
    F, y = lin_feats(c), (np.roll(c, -H) - c) / 50
    m = tr_mask & ~np.isnan(F).any(axis=1) & ~np.isnan(y)
    m[-H:] = False
    Fs.append(F[m]), Ys.append(y[m])
Fa, Ya = np.concatenate(Fs), np.concatenate(Ys)
beta = np.linalg.solve(Fa.T @ Fa + 1.0 * np.eye(Fa.shape[1]), Fa.T @ Ya)   # 릿지 회귀


def resid_score(c):
    """실제 CO₂ - (1시간 전 값 + 1시간 전에 한 예측), 15분 평균"""
    pred_change = lin_feats(c) @ beta * 50
    expected = np.roll(c + pred_change, H)
    expected[:H] = np.nan
    return pd.Series(c - expected).rolling(3).mean().values


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


# 공간별 검증 점수 분위수로 임계값 후보를 만들고, 결합 헛경보가 검증에서 하루 1회 이하인 것 중 가장 낮은 임계값 선택
val_scores = {r: ae_score(co2[r].values, tmp[r].values)[0] for r in rooms}
val_resid = {r: resid_score(co2[r].values) for r in rooms}
QS = [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]


def thr(r, q, src=None):
    v = (val_scores if src is None else src)[r][(idx >= TRAIN_END) & (idx < VAL_END)]
    return np.nanquantile(v, q)


_cache = {}


def scores(c, t):
    """같은 데이터로 여러 번 부르므로 점수를 저장해 둠"""
    key = (c.tobytes(), t.tobytes())
    if key not in _cache:
        s, cause = ae_score(c, t)
        _cache[key] = (np.nan_to_num(s), cause, np.nan_to_num(resid_score(c), nan=-1e9))
    return _cache[key]


def model_fn(q, qr=None):
    """오토인코더 복원 오차(분위수 q) 또는 예측 잔차(분위수 qr)가 임계값을 넘으면 경보"""
    qr = q if qr is None else qr
    def f(r, c, t):
        s, cause, rs = scores(c, t)
        res = rs > thr(r, qr, val_resid)
        return (s > thr(r, q)) | res, np.where(res, "co2", cause)
    return f


def comb_fn(q, qr=None):
    def f(r, c, t):
        a1, c1 = model_fn(q, qr)(r, c, t)
        a2, c2 = rule_alarm(c, t)
        return a1 | a2, np.where(a2, c2, c1)
    return f


rule_fn = lambda r, c, t: rule_alarm(c, t)

# 어떤 탐지기가 검증 기간(고장 없음)에 헛경보를 내는지 공간별로 셈 (임계값 조정용)
diag = {}
vdays = (pd.Timestamp(VAL_END) - pd.Timestamp(TRAIN_END)).days
for r in rooms:
    c, t = co2[r].values, tmp[r].values
    comps = rule_alarm(c, t, parts=True)
    comps["오토인코더(0.999)"] = np.nan_to_num(ae_score(c, t)[0]) > thr(r, 0.999)
    comps["예측잔차(0.999)"] = np.nan_to_num(resid_score(c), nan=-1e9) > thr(r, 0.999, val_resid)
    diag[r] = {k: round(len(events(v, np.full(len(c), ""), r, TRAIN_END, VAL_END)) / vdays, 2) for k, v in comps.items()}
print("\n[검증 기간 탐지기별 헛경보/일]\n", pd.DataFrame(diag).to_string())
print("\n[검증 기간 헛경보/일, 공간 합]  규칙만:", round(val_fa(rule_fn), 2))
# 검증 기간(고장 심음)에서 오토인코더·예측 잔차 분위수 조합별 탐지율과 헛경보
grid = []
for qa in QS:
    for qr in QS:
        evs = []
        for r in rooms:
            c, t = data_val[r]
            a, cause = comb_fn(qa, qr)(r, c, t)
            evs += events(a, cause, r, TRAIN_END, VAL_END)
        per, fa = evaluate(evs, labels_val, vdays)
        grid.append([qa, qr, round(per["탐지"].mean() * 100), round(fa, 2)])
grid = pd.DataFrame(grid, columns=["AE분위수", "잔차분위수", "검증탐지율%", "검증헛경보/일"])
ok = grid[grid["검증헛경보/일"] <= 1]
if len(ok) == 0:
    print("  ※ 어떤 조합도 검증 헛경보 1회/일 이하가 안 됨 → 가장 엄격한 조합 사용")
    ok = grid.tail(1)
best = ok.sort_values(["검증탐지율%", "검증헛경보/일"], ascending=[False, True]).iloc[0]
print("\n[검증 기간 임계값 조합 상위 8개 (헛경보 1회/일 이하)]\n",
      ok.sort_values(["검증탐지율%", "검증헛경보/일"], ascending=[False, True]).head(8).to_string(index=False))
q_best, qr_best = float(best["AE분위수"]), float(best["잔차분위수"])
print(f"→ 선택: 오토인코더 {q_best}, 예측 잔차 {qr_best}")


def run_test(fn):
    evs = []
    for r in rooms:
        c, t = data[r]
        a, cause = fn(r, c, t)
        evs += events(a, cause, r, VAL_END, TEST_END)
    return evs


summary = []
for name, fn in [("모델", model_fn(q_best, qr_best)), ("규칙", rule_fn), ("결합", comb_fn(q_best, qr_best))]:
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

# 임계값을 바꿔 가며 탐지율–헛경보 (보고용. 임계값 선택은 위에서 검증 기간으로 이미 끝남)
sweep = []
for q in QS:
    per, fa = evaluate(run_test(comb_fn(q)), labels)
    sweep.append([q, round(per["탐지"].mean() * 100), round(fa, 2)])
print("\n[임계값별 결합 성적, 시험]\n",
      pd.DataFrame(sweep, columns=["분위수", "탐지율%", "헛경보/일"]).to_string(index=False))
