import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 과제 5 — 상태 추정 (사용 여부 / 환기 선제 경보)
# v3: 사용 여부 MLP를 state_mlp.pt로 저장 (과제 7 엣지 변환용, 결과는 v2와 같음)
# v2: 환기 경보에 앞으로 30분 추정 시간표를 입력으로 추가해 일정 포함/제외 비교
# 실제 시간표·예약·관찰 기록이 없어서 데이터 흐름으로 대신 만듦 (보고서에 가정으로 명시):
#   "관찰 대체" 정답 : CO₂ 흐름을 앞뒤로 보고 판단한 사용 여부 (사람이 나중에 그래프를 보고 표시하는 것과 비슷)
#   "추정 시간표"     : 학습 기간의 관찰 대체 정답을 요일·30분 칸별로 모아, 절반 넘게 쓰인 칸을 "수업/예약"으로 봄
#   → README대로 "일정 기반 정답"으로 학습하고, "관찰 대체" 정답으로 평가
# 먼저 clean.py co2, clean.py temperature를 실행해 둘 것
DIR = "/home/prettygeun/ts-data/"
TRAIN_END, VAL_END, TEST_END = "2026-09-21", "2026-10-05", "2026-10-19"
HOLIDAYS = ["2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05", "2026-10-09"]
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

co2 = pd.read_pickle(DIR + "co2_room_5min.pkl")
tmp = pd.read_pickle(DIR + "temperature_room_5min.pkl")
rooms = list(co2.columns)
idx = co2.index
offday = (idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin(HOLIDAYS)
split = np.where(idx < TRAIN_END, "학습", np.where(idx < VAL_END, "검증", "시험"))


# ---------- 1. 정답 만들기 ----------
def observed(c):
    """관찰 대체: 앞뒤 30분 흐름으로 본 사용 여부 (미래 값을 쓰므로 정답 전용, 입력으로는 못 씀)"""
    cs = pd.Series(c).rolling(3, center=True, min_periods=1).mean()
    base = cs.rolling(288, center=True, min_periods=36).quantile(0.05)     # 그날 바닥값
    rise = cs.shift(-3) - cs.shift(3)                                     # 앞뒤 30분 변화
    occ = (rise > 20) | ((cs - base > 250) & (rise > -30))               # 오르는 중, 또는 높게 유지
    occ = occ.astype(float).rolling(3, center=True, min_periods=1).median() > 0.5
    return occ.values & ~np.isnan(c)


obs = {r: observed(co2[r].values) for r in rooms}

# 추정 시간표: 학습 기간 평일의 요일 × 30분 칸별 사용 비율 ≥ 50%
slot = idx.hour * 2 + idx.minute // 30
sched = {}
for r in rooms:
    df = pd.DataFrame({"요일": idx.dayofweek, "칸": slot, "사용": obs[r]})[(split == "학습") & ~offday]
    rate = df.groupby(["요일", "칸"])["사용"].mean()
    table = rate.reindex(pd.MultiIndex.from_arrays([idx.dayofweek, slot])).fillna(0).values
    sched[r] = (table >= 0.5) & ~offday
    print(f"[추정 시간표] {r}: 평일 하루 평균 {sched[r][~offday].mean() * 24:.1f}시간 사용")
    # 일정 기반 정답이 관찰과 얼마나 다른지 (README: 먼저 확인할 것)
    m = (split == "시험") & ~np.isnan(co2[r].values)
    o, s = obs[r][m], sched[r][m]
    print(f"   시험 기간 일정 vs 관찰: 일치 {np.mean(o == s) * 100:.1f}%, "
          f"일정엔 있는데 안 씀(노쇼 등) {np.mean(s & ~o) * 100:.1f}%, 일정 없이 씀 {np.mean(~s & o) * 100:.1f}%")


# ---------- 공통: 분류기 학습·평가 ----------
class Net(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.f = nn.Linear(n_in, 1) if hidden == 0 else nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit(X, y, hidden, pos_weight=1.0, epochs=30):
    """hidden=0이면 로지스틱 회귀, 아니면 작은 신경망(MLP)"""
    torch.manual_seed(0)
    model = Net(X.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    Xt, yt = torch.tensor(X, dtype=torch.float32, device=dev), torch.tensor(y, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(model(Xt[b]), yt[b]).backward()
            opt.step()
    return model


def prob(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=dev))).cpu().numpy()


def metrics(y, p):
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-9)
    rec0, pre0 = tn / max(tn + fp, 1), tn / max(tn + fn, 1)
    f1_0 = 2 * pre0 * rec0 / max(pre0 + rec0, 1e-9)
    return {"정확도": (tp + tn) / len(y), "재현율": rec, "정밀도": pre, "F1": f1, "macro-F1": (f1 + f1_0) / 2,
            "빈방→사용 오류율": fp / max(fp + tn, 1), "혼동[TN FP / FN TP]": f"{tn} {fp} / {fn} {tp}"}


def report(title, rows):
    df = pd.DataFrame(rows).set_index("방법")
    for c in ["정확도", "재현율", "정밀도", "F1", "macro-F1", "빈방→사용 오류율"]:
        df[c] = (df[c] * 100).round(1)
    print(f"\n{title}\n", df.to_string())


def lag(c, k):
    out = np.roll(c, k)
    out[:k] = np.nan
    return out


mc, sc = np.nanmean(co2[idx < TRAIN_END].values), np.nanstd(co2[idx < TRAIN_END].values)


# ---------- 2. 사용 여부 추정 ----------
# 입력: 센서 값만 (지난 1시간 CO₂ 변화, 지난 24시간 최저값 대비 높이, 온도 변화, 공간).
# 일정·시각은 넣지 않음 — 정답을 일정에서 만들었으므로 넣으면 모델이 일정을 베껴서 센서로 판단하는 법을 배우지 않음.
def occ_features(r):
    c, t = co2[r].values, tmp[r].values
    base24 = pd.Series(c).rolling(288, min_periods=36).min().values
    onehot = np.tile(np.eye(len(rooms))[rooms.index(r)], (len(c), 1))
    F = np.column_stack([(c - lag(c, k)) / 50 for k in (1, 2, 3, 6, 12)] +
                        [(c - base24) / 100, (c - mc) / sc, t - lag(t, 3), t - lag(t, 12), onehot])
    return F


print("\n========== 사용 여부 추정 ==========")
F_all = {r: occ_features(r) for r in rooms}


def gather(sp, label):
    X, y, o, R, T = [], [], [], [], []
    for r in rooms:
        m = (split == sp) & ~np.isnan(F_all[r]).any(axis=1)
        X.append(F_all[r][m])
        y.append(label[r][m])
        o.append(obs[r][m])
        R += [r] * int(m.sum())
        T.append(idx[m])
    return np.concatenate(X), np.concatenate(y).astype(float), np.concatenate(o).astype(int), np.array(R), np.concatenate(T)


Xtr, ytr, _, _, _ = gather("학습", sched)
Xva, yva, ova, _, _ = gather("검증", sched)
Xte, yte, ote, Rte, Tte = gather("시험", sched)
print(f"학습 {len(Xtr)}칸 (일정상 사용 {ytr.mean() * 100:.0f}%), 시험 {len(Xte)}칸 (관찰 대체상 사용 {ote.mean() * 100:.0f}%)")

rows, preds = [], {}
rule = (Xte[:, 2] * 50 >= 30).astype(int)                    # 규칙: 최근 15분 CO₂ +30ppm 이상이면 사용중
rows.append({"방법": "규칙(15분 +30ppm)", **metrics(ote, rule)})
for name, hidden in [("로지스틱 회귀", 0), ("MLP", 32)]:
    model = fit(Xtr, ytr, hidden)
    # 판단 기준(0.5 대신)은 검증 기간 관찰 대체 정답 기준 F1이 가장 높은 값으로
    pv = prob(model, Xva)
    th = max(np.arange(0.1, 0.91, 0.05), key=lambda h: metrics(ova, (pv >= h).astype(int))["F1"])
    p = (prob(model, Xte) >= th).astype(int)
    preds[name] = p
    if name == "MLP":                                         # 과제 7(엣지 변환)용 저장
        torch.save(model.state_dict(), DIR + "state_mlp.pt")
        np.save(DIR + "state_test_X.npy", Xte.astype(np.float32))
        json.dump({"n_in": int(Xte.shape[1]), "hidden": 32, "threshold": float(th)}, open(DIR + "state_mlp.json", "w"))
    rows.append({"방법": f"{name} (기준 {th:.2f})", **metrics(ote, p)})
    rows.append({"방법": f"  └ 일정 기반 정답으로 채점", **metrics(yte.astype(int), p)})
report("[시험] 사용 여부 — 관찰 대체 정답 기준 (%)", rows)

best = max(preds, key=lambda k: metrics(ote, preds[k])["F1"])
err = pd.DataFrame({"공간": Rte, "시각": pd.DatetimeIndex(Tte).hour, "틀림": preds[best] != ote,
                    "종류": np.where((preds[best] == 1) & (ote == 0), "빈방→사용", np.where((preds[best] == 0) & (ote == 1), "사용→빈방", ""))})
print(f"\n[가장 많이 틀리는 상황 — {best}] 공간·시각별 틀린 칸 수 상위 8")
print(err[err["틀림"]].groupby(["공간", "시각", "종류"]).size().sort_values(ascending=False).head(8).to_string())


# ---------- 3. 환기 선제 경보 ----------
# 양성: 지금 1,000ppm 미만인데 30분(6칸) 안에 1,000ppm 이상이 됨. 이미 넘은 시점은 샘플에서 뺌
# (이미 넘었으면 "선제"가 아니고, 넣으면 "지금 높다"만 보고 맞히는 쉬운 문제가 섞여 성적이 부풀려짐)
print("\n========== 환기 선제 경보 ==========")
USE_SCHED = True
hour = idx.hour + idx.minute / 60
tf = np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), offday.astype(float)])


def vent_data(sp):
    X, y = [], []
    for r in rooms:
        c = co2[r].values
        fut = np.column_stack([np.roll(c, -k) for k in range(1, 7)])
        fut[-6:] = np.nan
        base24 = pd.Series(c).rolling(288, min_periods=36).min().values
        # v2: 앞으로 30분의 일정(추정 시간표)도 입력 — 일정은 미리 알 수 있는 값이고, 환기 경보 정답은 일정이 아니라 CO₂에서 만들었으므로 베끼기 문제가 없음
        sch = sched[r].astype(float)
        fsch = np.column_stack([np.roll(sch, -k) for k in range(0, 7)]) if USE_SCHED else np.zeros((len(c), 0))
        F = np.column_stack([(c - lag(c, k)) / 50 for k in range(1, 13)] +
                            [(c - mc) / sc, (c - base24) / 100, tf, np.tile(np.eye(len(rooms))[rooms.index(r)], (len(c), 1)), fsch])
        m = (split == sp) & (c < 1000) & ~np.isnan(F).any(axis=1) & ~np.isnan(fut).any(axis=1)
        X.append(F[m])
        y.append((np.nanmax(fut[m], axis=1) >= 1000).astype(int))
    return np.concatenate(X), np.concatenate(y)


USE_SCHED = False
Vtr0, vtr = vent_data("학습")
Vva0, vva = vent_data("검증")
Vte0, vte = vent_data("시험")
USE_SCHED = True
Vtr, _ = vent_data("학습")
Vva, _ = vent_data("검증")
Vte, _ = vent_data("시험")
print(f"양성 비율: 학습 {vtr.mean() * 100:.1f}%, 검증 {vva.mean() * 100:.1f}%, 시험 {vte.mean() * 100:.1f}% "
      f"(시험 양성 {vte.sum()}건)")

rows = [{"방법": "전부 '정상'이라고 답함", **metrics(vte, np.zeros_like(vte))}]   # 정확도가 의미 없음을 보이기 위함
level = Vte[:, 12] * sc + mc
rows.append({"방법": "규칙(지금 850ppm 이상)", **metrics(vte, (level >= 850).astype(int))})
pw = (1 - vtr.mean()) / max(vtr.mean(), 1e-6)                  # 불균형: 양성에 가중치
for name, hidden in [("로지스틱 회귀", 0), ("MLP", 32)]:
    for tag, (A, B, C) in [("일정 없음", (Vtr0, Vva0, Vte0)), ("일정 포함", (Vtr, Vva, Vte))]:
        model = fit(A, vtr, hidden, pos_weight=pw)
        pv = prob(model, B)
        # 기준: 검증 재현율 85% 이상을 유지하는 가장 높은 값 (헛경보를 최대한 줄임)
        cands = [h for h in np.arange(0.05, 0.996, 0.005) if metrics(vva, (pv >= h).astype(int))["재현율"] >= 0.85]
        th = max(cands) if cands else 0.05
        rows.append({"방법": f"{name}·{tag} (기준 {th:.3f})", **metrics(vte, (prob(model, C) >= th).astype(int))})
report("[시험] 환기 선제 경보 (%)", rows)
