import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 과제 5 보완 — 환기 선제 경보를 과제 3 GRU 예측으로 판단
# 경보: 앞으로 30분(6칸) 예측값 중 최대가 기준 T 이상이면 "곧 1,000ppm 초과"
# T는 검증 기간에서 재현율 85% 이상을 유지하는 가장 높은 값으로 정함. 시험 기간은 평가에만 사용.
# 비교하는 GRU 두 개 (둘 다 과제 3에서 이미 저장됨, 일정 없음 모델):
#   gru_co2.pt       : 학습 기간(7/27~9/20)만으로 학습 → 검증 기간으로 T를 고르는 게 깨끗함
#   gru_co2_final.pt : 학습+검증(7/27~10/4)으로 재학습 → 9/21 이후 바뀐 시기를 봤지만, 검증 기간을 이미 학습에 써서
#                      T를 검증으로 고르면 약간 낙관적일 수 있음 (보고서에 명시)
DIR = "/home/prettygeun/ts-data/"
TRAIN_END, VAL_END, TEST_END = "2026-09-21", "2026-10-05", "2026-10-19"
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open(DIR + "gru_co2_norm.json"))
mu, sd, DS, rooms = cfg["mu"], cfg["sd"], cfg["dscale"], cfg["rooms"]
HOLIDAYS = cfg["holidays"]
LB, H = cfg["lookback"], cfg["horizon"]

room = pd.read_pickle(DIR + "co2_room_5min.pkl")
idx = room.index
hour = idx.hour + idx.minute / 60
time_feat = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                      ((idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin(HOLIDAYS)).astype(float)], axis=1)


class GRU(nn.Module):           # 과제 3과 같은 구조
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def load(path):
    state = torch.load(path, map_location=dev)
    n_in = state["gru.weight_ih_l0"].shape[1]       # 일정 칸(12개, 일정 없음이면 0으로 채움)이 있는 버전인지 자동 판단
    m = GRU(n_in).to(dev)
    m.load_state_dict(state)
    m.eval()
    return m, n_in


def samples(r, start, end, n_in):
    """t가 [start, end)이고 지금 1,000ppm 미만인 시점: GRU 입력, 정답(30분 안 1,000 이상), 지금 값"""
    c = room[r].values
    z = (c - mu) / sd
    onehot = np.eye(len(rooms))[rooms.index(r)]
    X, y, now = [], [], []
    for t in np.where((idx >= start) & (idx < end))[0]:
        if t < LB - 1 or t + 6 >= len(c) or not c[t] < 1000:
            continue
        win, fut = z[t - LB + 1:t + 1], c[t + 1:t + 7]
        if np.isnan(win).any() or np.isnan(fut).any():
            continue
        rel = (win - win[-1]) * sd / DS
        cols = [rel, time_feat[t - LB + 1:t + 1], np.tile(onehot, (LB, 1))]
        if n_in > 1 + 3 + len(rooms):
            cols.append(np.zeros((LB, n_in - 1 - 3 - len(rooms))))
        X.append(np.column_stack(cols))
        y.append(int(fut.max() >= 1000))
        now.append(c[t])
    return np.array(X, np.float32), np.array(y), np.array(now)


def max30(model, X, now):
    """앞으로 30분 예측값의 최대(ppm)"""
    with torch.no_grad():
        d = np.concatenate([model(torch.from_numpy(X[i:i + 4096]).to(dev)).cpu().numpy()
                            for i in range(0, len(X), 4096)]) * DS
    return (now[:, None] + d[:, :6]).max(axis=1)


def metrics(y, p):
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    rec, pre = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-9)
    rec0, pre0 = tn / max(tn + fp, 1), tn / max(tn + fn, 1)
    f1_0 = 2 * pre0 * rec0 / max(pre0 + rec0, 1e-9)
    return {"정확도": (tp + tn) / len(y), "재현율": rec, "정밀도": pre, "F1": f1, "macro-F1": (f1 + f1_0) / 2,
            "하루당 헛경보(공간 합)": fp / 14, "혼동[TN FP / FN TP]": f"{tn} {fp} / {fn} {tp}"}


def gather(start, end, n_in):
    parts = [samples(r, start, end, n_in) for r in rooms]
    return [np.concatenate([p[i] for p in parts]) for i in range(3)]


rows = []
for name, path in [("GRU(학습만)", "gru_co2.pt"), ("GRU(학습+검증)", "gru_co2_final.pt")]:
    model, n_in = load(DIR + path)
    Xv, yv, nv = gather(TRAIN_END, VAL_END, n_in)
    Xt, yt, nt = gather(VAL_END, TEST_END, n_in)
    mv, mt = max30(model, Xv, nv), max30(model, Xt, nt)
    if not rows:
        print(f"시험 샘플 {len(yt)}개, 양성 {yt.sum()}건 ({yt.mean() * 100:.1f}%)")
        rows.append({"방법": "규칙(지금 850ppm 이상)", **metrics(yt, (nt >= 850).astype(int))})
    rows.append({"방법": f"{name}·기준 1000 그대로", **metrics(yt, (mt >= 1000).astype(int))})
    cands = [T for T in range(700, 1001, 5) if metrics(yv, (mv >= T).astype(int))["재현율"] >= 0.85]
    T = max(cands) if cands else 700
    rows.append({"방법": f"{name}·기준 {T} (검증 재현율 85%)", **metrics(yt, (mt >= T).astype(int))})

df = pd.DataFrame(rows).set_index("방법")
for c in ["정확도", "재현율", "정밀도", "F1", "macro-F1"]:
    df[c] = (df[c] * 100).round(1)
df["하루당 헛경보(공간 합)"] = df["하루당 헛경보(공간 합)"].round(1)
pd.set_option("display.width", 250)
print("\n[시험] 환기 선제 경보 — GRU 예측 기반 (%)\n", df.to_string())
print("\n참고: 분류기 결과(state.py)는 로지스틱 macro-F1 57.1, MLP 56.3 / 규칙 60.9")
