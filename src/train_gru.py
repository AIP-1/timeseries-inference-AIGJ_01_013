import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 사용법: python train_gru.py [항목] [일정]        → 학습(7/27~9/20) + 검증 결과, 가장 좋은 epoch 기록
#         python train_gru.py final [항목] [일정]  → 학습+검증 기간(7/27~10/4)으로 같은 epoch 수만큼 다시 학습 후 시험 평가 (한 번만)
#   [항목] co2(기본) / temperature / humidity — 먼저 clean.py, baseline.py를 같은 항목으로 실행해 둘 것
#   [일정] none   : 일정 없음 (기본, v3와 같음)
#          fixed  : 가정한 고정 시간표 — 평일(공휴일 제외) 09~12시, 13~18시 사용
#          oracle : (CO₂만) 데이터에서 복원한 "사용 중" 표시 (15분 뒤 CO₂가 20ppm 넘게 오르면 사용 중). 실제 예약 기록이 있다면 얻을 수 있는 상한 — 미래 CO₂로 만든 값이라 실제 운영에선 쓸 수 없음
# v3: 9/21 이후 CO₂ 수준이 크게 낮아져서, 절대값 대신 "지금 값 대비 차이"만 입력으로 씀
# v4: 앞으로 60분(12칸)의 일정을 입력에 추가 (일정은 미리 알 수 있는 값이라 누수 아님)
DIR = "/home/prettygeun/ts-data/"
LOOKBACK, HORIZON = 24, 12           # 지난 2시간 → 5~60분 뒤
TRAIN_END, VAL_END, TEST_END = "2026-09-21", "2026-10-05", "2026-10-19"
# 공휴일(대체공휴일 포함, 학교 일정과 다르면 고칠 것): 광복절 대체 8/17, 추석 9/24~26, 개천절 대체 10/5, 한글날 10/9
HOLIDAYS = ["2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05", "2026-10-09"]
torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
args = sys.argv[1:]
MEAS = next((a for a in args if a in ("co2", "temperature", "humidity")), "co2")
SCHED = next((a for a in args if a in ("none", "fixed", "oracle")), "none")
assert not (SCHED == "oracle" and MEAS != "co2"), "oracle 일정은 CO₂에만 사용"
DSCALE, UNIT = {"co2": (50.0, "ppm"), "temperature": (0.5, "℃"), "humidity": (2.0, "%")}[MEAS]   # 정답(변화량) 학습 단위
TAG = "" if SCHED == "none" else "_" + SCHED          # 모델 파일 이름 구분
EP_FILE = "eval_points.csv" if MEAS == "co2" else f"eval_points_{MEAS}.csv"

room = pd.read_pickle(DIR + f"{MEAS}_room_5min.pkl")
rooms = list(room.columns)
idx = room.index

# 정규화 값은 학습 기간에서만 계산
train_vals = room[idx < TRAIN_END].values
mu, sd = float(np.nanmean(train_vals)), float(np.nanstd(train_vals))

# 시각 특징: 미래 시각도 미리 알 수 있는 값이라 누수 아님
hour = idx.hour + idx.minute / 60
time_feat = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                      ((idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin(HOLIDAYS)).astype(float)], axis=1)

offday = (idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin(HOLIDAYS)


def schedule(r):
    """공간 r의 5분 칸별 일정(1=사용, 0=비어 있음)"""
    if SCHED == "fixed":
        h = idx.hour
        return (~offday & (((h >= 9) & (h < 12)) | ((h >= 13) & (h < 18)))).astype(float)
    if SCHED == "oracle":
        y = room[r]
        return ((y.shift(-3) - y) > 20).astype(float).values
    return np.zeros(len(idx))


def make_samples(r, start, end):
    """공간 r에서 t가 [start, end)인 샘플. 입력 t-23..t, 정답 t+1..t+12 - y(t)"""
    y = room[r].values
    z = (y - mu) / sd
    onehot = np.zeros(len(rooms))
    onehot[rooms.index(r)] = 1
    sch = schedule(r)
    X, Y, T = [], [], []
    ts = np.where((idx >= start) & (idx < end))[0]
    for t in ts:
        if t < LOOKBACK - 1 or t + HORIZON >= len(y):
            continue
        win, fut = z[t - LOOKBACK + 1:t + 1], z[t + 1:t + 1 + HORIZON]
        if np.isnan(win).any() or np.isnan(fut).any():
            continue
        rel = (win - win[-1]) * sd / DSCALE          # 지금 값 대비 차이 (수준 변화에 덜 민감)
        future_sch = sch[t + 1:t + 1 + HORIZON]          # 앞으로 60분 일정
        feat = np.column_stack([rel, time_feat[t - LOOKBACK + 1:t + 1], np.tile(onehot, (LOOKBACK, 1)),
                                np.tile(future_sch, (LOOKBACK, 1))])
        X.append(feat)
        Y.append((fut - win[-1]) * sd / DSCALE)   # 지금 값에서의 변화량
        T.append(t)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(T)


def build(start, end):
    parts = [make_samples(r, start, end) + (r,) for r in rooms]
    return parts


class GRU(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(X[i:i + 4096]).to(dev)).cpu().numpy()
                               for i in range(0, len(X), 4096)])


def evaluate(model, parts, split):
    """과제 2와 같은 평가 시점(eval_points)에서 모델과 직전값 MAE 비교"""
    ep = pd.read_csv(DIR + EP_FILE)
    ep["시각"] = pd.to_datetime(ep["시각"], utc=True).dt.tz_convert("Asia/Seoul")
    rows = []
    for X, Y, T, r in parts:
        pred = predict(model, X) * DSCALE   # 변화량(원 단위)
        true = Y * DSCALE
        times = idx[T]
        for hn, h in {"5분": 1, "15분": 3, "30분": 6, "60분": 12}.items():
            pts = set(ep[(ep["공간"] == r) & (ep["구간"] == split) & (ep["예측거리"] == hn)]["시각"])
            m = np.array([t in pts for t in times])
            rows.append([r, hn, int(m.sum()), float(np.abs(true[m, h - 1]).mean()),
                         float(np.abs(pred[m, h - 1] - true[m, h - 1]).mean())])
    res = pd.DataFrame(rows, columns=["공간", "예측거리", "평가점수", "직전값", "GRU"]).round(2)
    res["개선%"] = ((1 - res["GRU"] / res["직전값"]) * 100).round(1)
    return res


def stack(parts):
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train(Xtr, Ytr, epochs, Xva=None, Yva=None, path=None):
    """Xva가 있으면 검증 60분 MAE가 가장 좋은 모델을 path에 저장하고 그 epoch를 돌려줌"""
    torch.manual_seed(0)
    model = GRU(Xtr.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    Xt, Yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
    best, best_ep, wait = 1e9, epochs, 0
    for ep_i in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), Yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if Xva is None:
            print(f"epoch {ep_i + 1:2d}")
            continue
        va_mae = np.abs(predict(model, Xva) - Yva)[:, -1].mean() * DSCALE   # 60분 검증 MAE
        print(f"epoch {ep_i + 1:2d}  검증 60분 MAE {va_mae:.2f} {UNIT}")
        if va_mae < best:
            best, best_ep, wait = va_mae, ep_i + 1, 0
            torch.save(model.state_dict(), path)
        else:
            wait += 1
            if wait >= 10:              # 10번 연속 좋아지지 않으면 멈춤
                break
    if Xva is None:
        torch.save(model.state_dict(), path)
    else:
        model.load_state_dict(torch.load(path, map_location=dev))
    return model, best_ep


if "final" in sys.argv[1:]:
    cfg = json.load(open(DIR + f"gru_{MEAS}{TAG}_norm.json"))
    X, Y = stack(build("2026-07-27", VAL_END))
    print(f"[{MEAS}, 일정 {SCHED}] 학습+검증 샘플 {len(X)}, {cfg['best_epoch']} epoch 학습, 장치 {dev}")
    model, _ = train(X, Y, cfg["best_epoch"], path=DIR + f"gru_{MEAS}{TAG}_final.pt")
    print(f"\n[시험] MAE ({UNIT})\n", evaluate(model, build(VAL_END, TEST_END), "시험").to_string(index=False))
    sys.exit()

tr, va = build("2026-07-27", TRAIN_END), build(TRAIN_END, VAL_END)
Xtr, Ytr = stack(tr)
Xva, Yva = stack(va)
print(f"[{MEAS}, 일정 {SCHED}] 학습 샘플 {len(Xtr)}, 검증 샘플 {len(Xva)}, 장치 {dev}")
model, best_ep = train(Xtr, Ytr, 60, Xva, Yva, DIR + f"gru_{MEAS}{TAG}.pt")
json.dump({"mu": mu, "sd": sd, "rooms": rooms, "lookback": LOOKBACK, "horizon": HORIZON,
           "dscale": DSCALE, "holidays": HOLIDAYS, "best_epoch": best_ep, "schedule": SCHED, "measurement": MEAS},
          open(DIR + f"gru_{MEAS}{TAG}_norm.json", "w"), ensure_ascii=False)
print(f"\n가장 좋은 epoch: {best_ep}")
print(f"\n[검증] MAE ({UNIT})\n", evaluate(model, va, "검증").to_string(index=False))
