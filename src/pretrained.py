import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# 과제 6 — 사전학습 모델(Chronos) 추가 학습 없이 시험 기간 CO₂ 예측, 기준선·과제 3 GRU와 비교
# 사용법: python pretrained.py [모델 ...]   (기본: chronos-bolt-tiny, chronos-bolt-small, chronos-2)
#   처음 실행 때 HuggingFace에서 모델을 내려받음 (인터넷 필요, ~/.cache/huggingface 에 저장)
# 평가 시점: 과제 2의 eval_points.csv 시험 구간 (기준선·GRU와 같은 시점)
# 예측 구간: 10%·90% 분위수 사이에 실제 값이 들어간 비율 (80%에 가까울수록 좋음)
DIR = "/home/prettygeun/ts-data/"
VAL_END, TEST_END = "2026-10-05", "2026-10-19"
CTX = 2016                      # 입력 길이: 지난 7일 (5분 × 2016)
H = 12                          # 60분
BATCH = 64
dev = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = sys.argv[1:] or ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small", "amazon/chronos-2"]
# 과제 3 결과 (시험, 60분 MAE ppm, GRU 학습+검증 재학습) — 비교표용
GRU60 = {"사무실": 36.7, "실습실": 25.2, "회의실": 60.4}

from chronos import BaseChronosPipeline   # noqa: E402  (pip install chronos-forecasting)

room = pd.read_pickle(DIR + "co2_room_5min.pkl")
rooms = list(room.columns)
idx = room.index
ep = pd.read_csv(DIR + "eval_points.csv")
ep["시각"] = pd.to_datetime(ep["시각"], utc=True).dt.tz_convert("Asia/Seoul")
ep = ep[ep["구간"] == "시험"]
HN = {"5분": 1, "15분": 3, "30분": 6, "60분": 12}


def origins(r):
    """공간 r의 평가 시점(예측을 하는 시각 t) 위치 — 네 예측거리 중 하나라도 평가하는 시점"""
    ts = pd.DatetimeIndex(sorted(set(ep.loc[ep["공간"] == r, "시각"])))
    return idx.get_indexer(ts)


def rss_mb():
    """프로세스 메모리(RSS, MB). 젯슨은 CPU·GPU가 메모리를 같이 씀"""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024


def forecast(pipe, ctxs):
    """ctxs: (B, CTX) 텐서(앞쪽 결측은 NaN) → (B, H, 3) 분위수 10·50·90%"""
    if type(pipe).__name__ == "Chronos2Pipeline":     # chronos-2는 (시계열 수, 변수 수, 길이) 3차원 입력
        ctxs = ctxs[:, None, :]
    q, _ = pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
    if isinstance(q, list):                  # chronos-2는 시계열마다 (변수 수, H, 분위수) 목록으로 돌려줌
        q = torch.stack([x.reshape(-1, H, 3)[0] for x in q])
    return q.float().cpu().numpy()


rows, cover_rows, speed = [], [], []
for name in MODELS:
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    m0 = rss_mb()
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(name, device_map=dev)
    n_par = sum(p.numel() for p in pipe.model.parameters()) / 1e6
    print(f"\n[{name}] 불러오기 {time.time() - t0:.1f}초, 파라미터 {n_par:.1f}M, 장치 {dev}")

    for r in rooms:
        c = room[r].values.astype(np.float32)
        pos = origins(r)
        preds = np.full((len(pos), H, 3), np.nan, np.float32)
        for i in range(0, len(pos), BATCH):
            p = pos[i:i + BATCH]
            X = np.full((len(p), CTX), np.nan, np.float32)
            for j, t in enumerate(p):
                w = c[max(0, t - CTX + 1):t + 1]
                X[j, CTX - len(w):] = w
            preds[i:i + len(p)] = forecast(pipe, torch.from_numpy(X))
        times = idx[pos]
        for hn, h in HN.items():
            pts = set(ep.loc[(ep["공간"] == r) & (ep["예측거리"] == hn), "시각"])
            m = np.array([t in pts for t in times])
            now, true = c[pos[m]], c[pos[m] + h]
            ok = ~np.isnan(true) & ~np.isnan(now)
            med, lo, hi = preds[m][ok, h - 1, 1], preds[m][ok, h - 1, 0], preds[m][ok, h - 1, 2]
            rows.append([name.split("/")[-1], r, hn, int(ok.sum()), float(np.abs(true[ok] - now[ok]).mean()),
                         float(np.abs(true[ok] - med).mean())])
            cover_rows.append([name.split("/")[-1], r, hn, float(((true[ok] >= lo) & (true[ok] <= hi)).mean() * 100),
                               float((hi - lo).mean())])

    # 1회 추론 시간 (시계열 1개, 60분 예측): 워밍업 5회 후 50회 반복
    x1 = torch.from_numpy(np.nan_to_num(room[rooms[0]].values[-CTX:].astype(np.float32), nan=400.0))[None]
    for _ in range(5):
        forecast(pipe, x1)
    lat = []
    for _ in range(50):
        if dev == "cuda":
            torch.cuda.synchronize()
        s = time.perf_counter()
        forecast(pipe, x1)
        if dev == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - s) * 1000)
    gpu_mb = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
    speed.append([name.split("/")[-1], round(n_par, 1), round(np.mean(lat), 1), round(np.percentile(lat, 95), 1),
                  round(gpu_mb, 0), round(rss_mb() - m0)])
    del pipe

res = pd.DataFrame(rows, columns=["모델", "공간", "예측거리", "평가점수", "직전값", "MAE"])
cov = pd.DataFrame(cover_rows, columns=["모델", "공간", "예측거리", "구간적중률%", "구간폭ppm"])
pd.set_option("display.width", 250)

t60 = res[res["예측거리"] == "60분"].pivot(index="공간", columns="모델", values="MAE").round(1)
t60.insert(0, "직전값", res[res["예측거리"] == "60분"].groupby("공간")["직전값"].first().round(1))
t60.insert(1, "GRU(과제3)", pd.Series(GRU60))
print("\n[시험] CO₂ 60분 MAE (ppm) — 기준선 / 직접 학습 / 사전학습")
print(t60.to_string())

print("\n[시험] 예측거리별 MAE (ppm)")
print(res.pivot_table(index=["모델", "공간"], columns="예측거리", values="MAE")[list(HN)].round(1).to_string())

print("\n[시험] 10~90% 예측 구간 적중률 (%) — 80%에 가까울수록 좋음")
print(cov.pivot_table(index=["모델", "공간"], columns="예측거리", values="구간적중률%")[list(HN)].round(1).to_string())
print("\n[시험] 60분 예측 구간 평균 폭 (ppm)")
print(cov[cov["예측거리"] == "60분"].pivot(index="공간", columns="모델", values="구간폭ppm").round(0).to_string())

print("\n[보드] 1회 추론(시계열 1개, 입력 7일 → 60분), 워밍업 5회 후 50회")
print(pd.DataFrame(speed, columns=["모델", "파라미터M", "평균ms", "p95ms", "GPU최대MB", "RSS증가MB"]).to_string(index=False))
print(f"측정 조건: 장치 {dev}, 전력 모드는 `sudo nvpmodel -q`로 확인해 함께 기록")

res.to_csv(DIR + "pretrained_result.csv", index=False)
cov.to_csv(DIR + "pretrained_interval.csv", index=False)
