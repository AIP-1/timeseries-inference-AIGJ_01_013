import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 사용법: python train_gru.py        → 학습 + 검증 결과
#         python train_gru.py test   → 저장된 모델로 시험 구간 평가 (마지막에 한 번만)
DIR = "/home/prettygeun/ts-data/"
LOOKBACK, HORIZON = 24, 12           # 지난 2시간 → 5~60분 뒤
TRAIN_END, VAL_END, TEST_END = "2026-09-21", "2026-10-05", "2026-10-19"
torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

room = pd.read_pickle(DIR + "co2_room_5min.pkl")
rooms = list(room.columns)
idx = room.index

# 정규화 값은 학습 기간에서만 계산
train_vals = room[idx < TRAIN_END].values
mu, sd = float(np.nanmean(train_vals)), float(np.nanstd(train_vals))

# 시각 특징: 미래 시각도 미리 알 수 있는 값이라 누수 아님
hour = idx.hour + idx.minute / 60
time_feat = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                      (idx.dayofweek >= 5).astype(float)], axis=1)


def make_samples(r, start, end):
    """공간 r에서 t가 [start, end)인 샘플. 입력 t-23..t, 정답 t+1..t+12 - y(t)"""
    y = room[r].values
    z = (y - mu) / sd
    onehot = np.zeros(len(rooms))
    onehot[rooms.index(r)] = 1
    X, Y, T = [], [], []
    ts = np.where((idx >= start) & (idx < end))[0]
    for t in ts:
        if t < LOOKBACK - 1 or t + HORIZON >= len(y):
            continue
        win, fut = z[t - LOOKBACK + 1:t + 1], z[t + 1:t + 1 + HORIZON]
        if np.isnan(win).any() or np.isnan(fut).any():
            continue
        feat = np.column_stack([win, time_feat[t - LOOKBACK + 1:t + 1], np.tile(onehot, (LOOKBACK, 1))])
        X.append(feat)
        Y.append(fut - win[-1])       # 지금 값에서의 변화량
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
    """과제 2와 같은 평가 시점(eval_points.csv)에서 모델과 직전값 MAE(ppm) 비교"""
    ep = pd.read_csv(DIR + "eval_points.csv")
    ep["시각"] = pd.to_datetime(ep["시각"], utc=True).dt.tz_convert("Asia/Seoul")
    rows = []
    for X, Y, T, r in parts:
        pred = predict(model, X) * sd       # 변화량(ppm)
        true = Y * sd
        times = idx[T]
        for hn, h in {"5분": 1, "15분": 3, "30분": 6, "60분": 12}.items():
            pts = set(ep[(ep["공간"] == r) & (ep["구간"] == split) & (ep["예측거리"] == hn)]["시각"])
            m = np.array([t in pts for t in times])
            rows.append([r, hn, int(m.sum()), float(np.abs(true[m, h - 1]).mean()),
                         float(np.abs(pred[m, h - 1] - true[m, h - 1]).mean())])
    res = pd.DataFrame(rows, columns=["공간", "예측거리", "평가점수", "직전값", "GRU"]).round(1)
    res["개선%"] = ((1 - res["GRU"] / res["직전값"]) * 100).round(1)
    return res


if len(sys.argv) > 1 and sys.argv[1] == "test":
    model = GRU(1 + 3 + len(rooms)).to(dev)
    model.load_state_dict(torch.load(DIR + "gru_co2.pt", map_location=dev))
    print("[시험] MAE (ppm)\n", evaluate(model, build(VAL_END, TEST_END), "시험").to_string(index=False))
    sys.exit()

tr = build("2026-07-27", TRAIN_END)
va = build(TRAIN_END, VAL_END)
Xtr = np.concatenate([p[0] for p in tr])
Ytr = np.concatenate([p[1] for p in tr])
Xva = np.concatenate([p[0] for p in va])
Yva = np.concatenate([p[1] for p in va])
print(f"학습 샘플 {len(Xtr)}, 검증 샘플 {len(Xva)}, 장치 {dev}")

model = GRU(Xtr.shape[2]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()
Xt, Yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
best, wait = 1e9, 0
for ep_i in range(60):
    model.train()
    perm = torch.randperm(len(Xt), device=dev)
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        opt.zero_grad()
        loss = loss_fn(model(Xt[b]), Yt[b])
        loss.backward()
        opt.step()
    va_mae = np.abs(predict(model, Xva) - Yva)[:, -1].mean() * sd   # 60분 검증 MAE
    print(f"epoch {ep_i + 1:2d}  검증 60분 MAE {va_mae:.1f} ppm")
    if va_mae < best:
        best, wait = va_mae, 0
        torch.save(model.state_dict(), DIR + "gru_co2.pt")
    else:
        wait += 1
        if wait >= 8:               # 8번 연속 좋아지지 않으면 멈춤
            break

json.dump({"mu": mu, "sd": sd, "rooms": rooms, "lookback": LOOKBACK, "horizon": HORIZON},
          open(DIR + "gru_co2_norm.json", "w"), ensure_ascii=False)
model.load_state_dict(torch.load(DIR + "gru_co2.pt", map_location=dev))
print("\n[검증] MAE (ppm)\n", evaluate(model, va, "검증").to_string(index=False))
