import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 과제 7 — 엣지 최적화: 모델 3종(예측 GRU / 이상 탐지 오토인코더 / 사용 여부 MLP)을
#   PyTorch CPU / PyTorch GPU / ONNX Runtime / TensorRT(FP16, 참고로 FP32도) 로 돌려 지연과 결과 차이를 비교
# 먼저 할 것: train_gru.py final, anomaly.py, state.py 를 실행해 모델 파일을 만들어 둘 것
#   (gru_co2_final.pt, ae_anomaly.pt, state_mlp.pt — state.py는 v3부터 MLP를 저장)
# 측정 조건: 실행 전에 전력 모드·클럭 고정 여부를 정하고 기록 (sudo nvpmodel -q, sudo jetson_clocks --show)
# 지연: 워밍업 WARM회는 빼고 REP회 반복의 평균·p95. GPU 방식은 입력이 이미 GPU에 있는 상태에서 잼(복사 시간 제외),
#       ONNX Runtime은 numpy 입력→출력까지(복사 포함)
# 차이: 기준 = PyTorch GPU(FP32). CO₂는 역정규화한 ppm, 분류는 판정이 바뀐 비율
DIR = "/home/prettygeun/ts-data/"
TRAIN_END, VAL_END, TEST_END = "2026-09-21", "2026-10-05", "2026-10-19"
WARM, REP = 20, 200
OUT = DIR + "edge/"
os.makedirs(OUT, exist_ok=True)
gpu = torch.cuda.is_available()
dev = "cuda" if gpu else "cpu"

try:
    import onnxruntime as ort
except ImportError:
    ort = None
try:
    import tensorrt as trt
except ImportError:
    trt = None

# ---------- 측정 중 보드 상태 기록 (tegrastats) ----------
tegra = None
if shutil.which("tegrastats"):
    tegra = subprocess.Popen(["tegrastats", "--interval", "500", "--logfile", OUT + "tegrastats.log"])

co2 = pd.read_pickle(DIR + "co2_room_5min.pkl")
tmp = pd.read_pickle(DIR + "temperature_room_5min.pkl")
rooms = list(co2.columns)
idx = co2.index
test = np.where((idx >= VAL_END) & (idx < TEST_END))[0]


# ---------- 모델 1: CO₂ 60분 예측 GRU (과제 3) ----------
class GRU(nn.Module):
    def __init__(self, n_in, horizon, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


cfg = json.load(open(DIR + "gru_co2_norm.json"))
LB, H, DS, mu, sd = cfg["lookback"], cfg["horizon"], cfg["dscale"], cfg["mu"], cfg["sd"]
state = torch.load(DIR + "gru_co2_final.pt", map_location="cpu")
n_in = state["gru.weight_ih_l0"].shape[1]
gru = GRU(n_in, H)
gru.load_state_dict(state)
hour = idx.hour + idx.minute / 60
offday = (idx.dayofweek >= 5) | idx.strftime("%Y-%m-%d").isin(cfg["holidays"])
time_feat = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), offday.astype(float)], axis=1)
Xg = []
for r in rooms:
    z = (co2[r].values - mu) / sd
    onehot = np.eye(len(rooms))[rooms.index(r)]
    for t in test:
        win = z[t - LB + 1:t + 1]
        if np.isnan(win).any():
            continue
        cols = [(win - win[-1]) * sd / DS, time_feat[t - LB + 1:t + 1], np.tile(onehot, (LB, 1))]
        if n_in > 1 + 3 + len(rooms):                         # 일정 칸이 있는 버전이면 0(일정 없음)으로 채움
            cols.append(np.zeros((LB, n_in - 1 - 3 - len(rooms))))
        Xg.append(np.column_stack(cols))
Xg = np.array(Xg, np.float32)

# ---------- 모델 2: 이상 탐지 오토인코더 (과제 4) ----------
W = 12


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(W * 4, 32), nn.ReLU(), nn.Linear(32, 8))
        self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, W * 4))

    def forward(self, x):
        return self.dec(self.enc(x)).view(-1, W, 4)


ae = AE()
ae.load_state_dict(torch.load(DIR + "ae_anomaly.pt", map_location="cpu"))
tr = idx < TRAIN_END
mc, sc = np.nanmean(co2[tr].values), np.nanstd(co2[tr].values)
mt, st = np.nanmean(tmp[rooms][tr].values), np.nanstd(tmp[rooms][tr].values)
Xa = []
for r in rooms:
    c, t = co2[r].values, tmp[r].values
    f = np.column_stack([(c - mc) / sc, (t - mt) / st, np.diff(c, prepend=np.nan) / 50, np.diff(t, prepend=np.nan) / 0.2])
    for i in test:
        w = f[i - W + 1:i + 1]
        if not np.isnan(w).any():
            Xa.append(w)
Xa = np.array(Xa, np.float32)

# ---------- 모델 3: 사용 여부 MLP (과제 5) ----------
scfg = json.load(open(DIR + "state_mlp.json"))


class Net(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


mlp = Net(scfg["n_in"], scfg["hidden"])
mlp.load_state_dict(torch.load(DIR + "state_mlp.pt", map_location="cpu"))
Xs = np.load(DIR + "state_test_X.npy").astype(np.float32)

MODELS = {"예측 GRU": (gru, Xg), "이상탐지 AE": (ae, Xa), "사용여부 MLP": (mlp, Xs)}
for k, (m, X) in MODELS.items():
    m.eval()
    print(f"{k}: 입력 {X.shape}, 파라미터 {sum(p.numel() for p in m.parameters()):,}개")


# ---------- 실행 방식 ----------
def run_torch(m, X, device, bs=1024):
    m = m.to(device)
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy() for i in range(0, len(X), bs)])


def time_it(fn, sync=False):
    for _ in range(WARM):
        fn()
    lat = []
    for _ in range(REP):
        if sync:
            torch.cuda.synchronize()
        s = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - s) * 1000)
    return np.mean(lat), np.percentile(lat, 95)


def export_onnx(m, X, path):
    m = m.cpu().eval()
    torch.onnx.export(m, torch.from_numpy(X[:4]), path, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17, dynamo=False)


class TRTModel:
    """ONNX → TensorRT 엔진. 입력·출력은 GPU torch 텐서 (pycuda 없이 torch 메모리 사용)"""

    def __init__(self, onnx_path, X, fp16, max_bs=1024):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(open(onnx_path, "rb").read()):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        conf = builder.create_builder_config()
        if fp16:
            conf.set_flag(trt.BuilderFlag.FP16)
        prof = builder.create_optimization_profile()
        shp = X.shape[1:]
        prof.set_shape("x", (1, *shp), (4, *shp), (max_bs, *shp))
        conf.add_optimization_profile(prof)
        t0 = time.time()
        plan = builder.build_serialized_network(network, conf)
        if plan is None:
            raise RuntimeError("엔진 생성 실패")
        self.build_s = time.time() - t0
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.max_bs = max_bs
        open(onnx_path.replace(".onnx", "_fp16.engine" if fp16 else "_fp32.engine"), "wb").write(bytes(plan))

    def __call__(self, x):
        """x: GPU float32 텐서 (배치, ...)"""
        self.ctx.set_input_shape("x", tuple(x.shape))
        y = torch.empty(tuple(self.ctx.get_tensor_shape("y")), device="cuda", dtype=torch.float32)
        self.ctx.set_tensor_address("x", x.data_ptr())
        self.ctx.set_tensor_address("y", y.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return y

    def run(self, X):
        out = []
        for i in range(0, len(X), self.max_bs):
            out.append(self(torch.from_numpy(X[i:i + self.max_bs]).cuda().contiguous()).cpu().numpy())
        torch.cuda.synchronize()
        return np.concatenate(out)


# ---------- 결과 차이 판단 ----------
def compare(name, ref, out):
    """기준(PyTorch GPU FP32) 대비 차이. GRU는 60분 예측 ppm, AE는 경보 판정, MLP는 사용 여부 판정"""
    if name == "예측 GRU":
        d = np.abs(out - ref) * DS                               # 변화량 예측 × 50 = ppm
        return f"CO₂ 최대 {d.max():.3f}ppm / 평균 {d.mean():.4f}ppm", d.max() <= 1.0
    if name == "이상탐지 AE":
        se = lambda o: ((o - MODELS[name][1]) ** 2).mean(axis=(1, 2))   # 복원 오차 = 이상 점수
        s_ref, s_out = se(ref), se(out)
        thr = np.quantile(s_ref, 0.999)                          # 과제 4와 같은 방식의 경보 기준(분위수 0.999)
        flip = ((s_ref > thr) != (s_out > thr)).mean() * 100
        return f"점수 최대 차이 {np.abs(s_out - s_ref).max():.2e}, 경보 판정 바뀜 {flip:.3f}%", flip <= 0.1
    pr = lambda o: 1 / (1 + np.exp(-o)) >= scfg["threshold"]
    flip = (pr(ref) != pr(out)).mean() * 100
    return f"사용 여부 판정 바뀜 {flip:.3f}% ({int((pr(ref) != pr(out)).sum())}/{len(ref)}건)", flip <= 0.1


lat_rows, diff_rows = [], []
for name, (m, X) in MODELS.items():
    print(f"\n===== {name} =====")
    ref = run_torch(m, X, dev)
    onnx_path = OUT + {"예측 GRU": "gru", "이상탐지 AE": "ae", "사용여부 MLP": "mlp"}[name] + ".onnx"
    ways = {}

    # 1) PyTorch CPU
    mc_ = m.cpu()
    for bs in (1, 4):
        xb = torch.from_numpy(X[:bs])
        with torch.no_grad():
            ways.setdefault("PyTorch CPU", {})[bs] = time_it(lambda: mc_(xb))
    diff_rows.append([name, "PyTorch CPU", *compare(name, ref, run_torch(m, X, "cpu"))])

    # 2) PyTorch GPU (기준)
    if gpu:
        mg = m.to("cuda")
        for bs in (1, 4):
            xb = torch.from_numpy(X[:bs]).cuda()
            with torch.no_grad():
                ways.setdefault("PyTorch GPU", {})[bs] = time_it(lambda: mg(xb), sync=True)
        diff_rows.append([name, "PyTorch GPU(기준)", "—", True])

    # 3) ONNX Runtime
    try:
        export_onnx(m, X, onnx_path)
        if ort is None:
            raise RuntimeError("onnxruntime 없음 (pip install onnxruntime)")
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        sess = ort.InferenceSession(onnx_path, providers=prov)
        label = f"ONNX Runtime({sess.get_providers()[0].replace('ExecutionProvider', '')})"
        for bs in (1, 4):
            xb = X[:bs]
            ways.setdefault(label, {})[bs] = time_it(lambda: sess.run(None, {"x": xb}))
        out = np.concatenate([sess.run(None, {"x": X[i:i + 1024]})[0] for i in range(0, len(X), 1024)])
        diff_rows.append([name, label, *compare(name, ref, out)])
    except Exception as e:
        print(f"  ONNX Runtime 실패: {e}")

    # 4) TensorRT FP16 (+ FP32 참고)
    if trt is None or not gpu or not os.path.exists(onnx_path):
        print("  TensorRT 건너뜀:", "tensorrt 모듈 없음" if trt is None else "GPU 또는 ONNX 없음")
        lat_rows += [[name, w, bs, round(v[bs][0], 3), round(v[bs][1], 3)] for w, v in ways.items() for bs in (1, 4)]
        continue
    for fp16 in (True, False):
        label = "TensorRT FP16" if fp16 else "TensorRT FP32(참고)"
        try:
            eng = TRTModel(onnx_path, X, fp16)
            print(f"  {label} 엔진 생성 {eng.build_s:.1f}초")
            for bs in (1, 4):
                xb = torch.from_numpy(X[:bs]).cuda().contiguous()
                ways.setdefault(label, {})[bs] = time_it(lambda: eng(xb), sync=True)
            diff_rows.append([name, label, *compare(name, ref, eng.run(X))])
        except Exception as e:
            print(f"  {label} 실패: {e}")
    lat_rows += [[name, w, bs, round(v[bs][0], 3), round(v[bs][1], 3)] for w, v in ways.items() for bs in (1, 4)]

if tegra:
    tegra.terminate()

pd.set_option("display.width", 250)
lat = pd.DataFrame(lat_rows, columns=["모델", "방식", "배치", "평균ms", "p95ms"])
print(f"\n[지연] 워밍업 {WARM}회 제외, {REP}회 반복 (ms)")
print(lat.pivot_table(index=["모델", "방식"], columns="배치", values=["평균ms", "p95ms"], sort=False).round(3).to_string())
diff = pd.DataFrame(diff_rows, columns=["모델", "방식", "기준(PyTorch GPU) 대비 차이", "기준 통과"])
print("\n[변환 전후 차이] 기준: CO₂ 1ppm 이하, 판정 변경 0.1% 이하")
print(diff.to_string(index=False))
lat.to_csv(OUT + "latency.csv", index=False)
diff.to_csv(OUT + "diff.csv", index=False)

# tegrastats 요약 (측정 전체 구간)
if tegra and os.path.exists(OUT + "tegrastats.log"):
    txt = open(OUT + "tegrastats.log").read()
    ram = [int(a) for a in re.findall(r"RAM (\d+)/\d+MB", txt)]
    gr3d = [int(a) for a in re.findall(r"GR3D_FREQ (\d+)%", txt)]
    vin = [int(a) for a in re.findall(r"VDD_IN (\d+)mW", txt)]
    print("\n[tegrastats, 측정 전체 구간]",
          f"RAM 최대 {max(ram)}MB" if ram else "", f"GPU 사용률 최대 {max(gr3d)}%" if gr3d else "",
          f"보드 전력 평균 {np.mean(vin):.0f}mW / 최대 {max(vin)}mW" if vin else "", f"(기록 {len(ram)}개, {OUT}tegrastats.log)")
else:
    print("\ntegrastats 없음 → 다른 터미널에서 `sudo tegrastats` 를 띄워 두고 값 기록")
print(f"장치: {torch.cuda.get_device_name(0) if gpu else 'CPU'}, torch {torch.__version__}, "
      f"onnxruntime {ort.__version__ if ort else '없음'}, TensorRT {trt.__version__ if trt else '없음'}")
