import json
import os
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
import pandas as pd

from collector import load_env
from features import ae_features, ae_window, clean_room, gru_input, occ_features, rule_alarm

# 과제 8 — 상시 추론 서비스 (매분 실행)
#   입력: collector.py가 쌓는 로컬 SQLite (최근 2일)
#   전처리: features.py — 학습(clean.py)과 같은 코드를 import
#   모델: 과제 7 결론대로 ONNX Runtime CPU (edge/gru.onnx, ae.onnx, mlp.onnx — edge.py가 만든 파일)
#   출력: MQTT 발행(retained) + 조회 API(HTTP, JSON) + 가동 로그(service_log.jsonl)
# 사용법: python service.py                 상시 실행
#         python service.py --summary       가동 로그 요약 (주기별 처리 시간, 경보 내역, 끊김 구간)
#         python service.py --once --now "2026-10-20 10:03" --no-mqtt   한 주기만 (시각 지정 재현 시험용)
# 정책:
#   - 막 시작된 5분 칸(1~4분치만 있음)은 쓰지 않음 → 마지막으로 "다 찬" 5분 칸까지가 입력 (학습 때와 같은 5분 평균이 되도록)
#   - 수집 끊김: 수집기가 3분 넘게 아무 메시지도 못 받음 / 데이터 끊김: 공간의 CO₂가 20분 넘게 안 들어옴(10분 간격 센서 고려)
#   - 경보 중복 억제: 같은 공간·같은 종류 경보는 처음 생길 때 1번, 이어지면 60분마다 1번 다시 알림,
#     15분 동안 다시 안 걸려야 "해제" 1번 (켜졌다 꺼졌다 반복하는 깜빡임 방지)
#   - 한 주기가 실패해도 프로세스는 계속, 다음 주기에 처음부터 다시 계산 (실패는 로그와 status 토픽에 남김)
DIR = os.path.expanduser("~/ts-data/")
VERSION = "service v1 (2026-10-21) / GRU 과제3 final · AE 과제4 · MLP 과제5 v3"
HISTORY = timedelta(days=2)                 # 사용 여부 특징이 24시간 최저값을 쓰므로 2일치
COLLECT_GAP_MIN, DATA_GAP_MIN, REMIND_MIN, CLEAR_MIN = 3, 20, 60, 15
VENT_WARN, VENT_ALARM = 860, 1000            # 과제 5 보완: 30분 내 최대 예측이 860 이상 "주의", 1000 이상 "환기 필요"
KST = timezone(timedelta(hours=9))
ASCII = {"사무실": "office", "실습실": "lab", "회의실": "meeting"}

args = sys.argv[1:]
env = load_env()
PREFIX = env.get("PUB_PREFIX", "ts-team4/inference")

# ---------- 모델·설정 불러오기 ----------
cfg = json.load(open(DIR + "gru_co2_norm.json"))
ROOMS, DS, H = cfg["rooms"], cfg["dscale"], cfg["horizon"]
so = ort.SessionOptions()
so.intra_op_num_threads = 2
S = {k: ort.InferenceSession(DIR + f"edge/{k}.onnx", so, providers=["CPUExecutionProvider"]) for k in ("gru", "ae", "mlp")}
N_IN = S["gru"].get_inputs()[0].shape[2]
MLP_TH = json.load(open(DIR + "state_mlp.json"))["threshold"]


def run(k, x):
    return S[k].run(None, {"x": x})[0]


def ae_score(w):
    """복원 오차(점수)와 원인 센서 (CO₂ 쪽/온도 쪽 중 오차가 큰 쪽)"""
    err = (run("ae", w[None]) - w[None]) ** 2
    return float(err.mean()), ("temp" if err[0, :, [1, 3]].sum() > err[0, :, [0, 2]].sum() else "co2")


def make_params():
    """학습 기간 정규화 값과 오토인코더 경보 기준(검증 기간 점수의 공간별 0.999 분위수, 과제 4와 같은 방식). 한 번 만들어 저장"""
    path = DIR + "service_params.json"
    if os.path.exists(path):
        return json.load(open(path))
    co2 = pd.read_pickle(DIR + "co2_room_5min.pkl")
    tmp = pd.read_pickle(DIR + "temperature_room_5min.pkl")
    idx = co2.index
    tr, va = idx < "2026-09-21", (idx >= "2026-09-21") & (idx < "2026-10-05")
    p = {"mc": float(np.nanmean(co2[tr].values)), "sc": float(np.nanstd(co2[tr].values)),
         "mt": float(np.nanmean(tmp[ROOMS][tr].values)), "st": float(np.nanstd(tmp[ROOMS][tr].values)), "ae_thr": {}}
    for r in ROOMS:
        f = ae_features(co2[r].values, tmp[r].values, p["mc"], p["sc"], p["mt"], p["st"])
        ws = [w for i in np.where(va)[0] if (w := ae_window(f, i)) is not None]
        err = ((run("ae", np.array(ws)) - np.array(ws)) ** 2).mean(axis=(1, 2))
        p["ae_thr"][r] = float(np.quantile(err, 0.999))
    json.dump(p, open(path, "w"), ensure_ascii=False, indent=1)
    return p


P = make_params()


# ---------- 입력 읽기 ----------
def read_raw(db, start, end):
    q = "SELECT time, device_id, location, meas, value FROM readings WHERE time >= ? AND time < ?"
    df = pd.read_sql_query(q, db, params=(start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()))
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601").dt.tz_convert("Asia/Seoul")
    return df


def last_readings(db, since, now):
    """공간별 CO₂ 마지막 수신 시각 (지금 채워지는 5분 칸 포함)"""
    q = "SELECT location, MAX(time) FROM readings WHERE meas = 'co2' AND time >= ? AND time <= ? GROUP BY location"
    rows = db.execute(q, (since.astimezone(timezone.utc).isoformat(), now.astimezone(timezone.utc).isoformat())).fetchall()
    return {loc: pd.Timestamp(t).tz_convert("Asia/Seoul") for loc, t in rows}


def heartbeat(db):
    row = db.execute("SELECT last_rx FROM heartbeat WHERE id = 1").fetchone()
    return datetime.fromisoformat(row[0]) if row else None


# ---------- 경보 관리 (중복 억제) ----------
active = {}      # (공간, 종류) → {"since", "last_sent", "detail"}


def update_alarms(now, current):
    """current: {(공간, 종류): 설명}. 새로 생김/계속(60분마다 재알림)/해제 이벤트 목록을 돌려줌"""
    events = []
    for key, detail in current.items():
        a = active.get(key)
        if a is not None:
            a.pop("clear_since", None)              # 해제 대기 중에 다시 걸리면 이어서 같은 경보
        if a is None:
            active[key] = {"since": now, "last_sent": now, "detail": detail}
            events.append({"상태": "발생", "공간": key[0], "종류": key[1], "설명": detail})
        elif now - a["last_sent"] >= timedelta(minutes=REMIND_MIN):
            a["last_sent"], a["detail"] = now, detail
            events.append({"상태": "계속", "공간": key[0], "종류": key[1], "설명": detail,
                           "시작": a["since"].isoformat(timespec="seconds")})
        else:
            a["detail"] = detail
    for key in [k for k in active if k not in current]:
        a = active[key]
        a.setdefault("clear_since", now)
        if now - a["clear_since"] < timedelta(minutes=CLEAR_MIN):
            continue
        active.pop(key)
        events.append({"상태": "해제", "공간": key[0], "종류": key[1], "시작": a["since"].isoformat(timespec="seconds"),
                       "지속분": round((a["clear_since"] - a["since"]).total_seconds() / 60)})
    for e in events:
        e["시각"] = now.isoformat(timespec="seconds")
    return events


# ---------- 한 주기 ----------
def cycle(db, now):
    end = pd.Timestamp(now).floor("5min")                     # 이 칸은 아직 채워지는 중 → 제외
    idx = pd.date_range(end - pd.Timedelta(HISTORY), end, freq="5min", inclusive="left")
    raw = read_raw(db, idx[0].to_pydatetime() - timedelta(minutes=10), end.to_pydatetime())
    rooms_co2 = clean_room(raw[raw["meas"] == "co2"], "co2", idx) if (raw["meas"] == "co2").any() else pd.DataFrame(index=idx)
    rooms_tmp = clean_room(raw[raw["meas"] == "temperature"], "temperature", idx) \
        if (raw["meas"] == "temperature").any() else pd.DataFrame(index=idx)
    i = len(idx) - 1
    current_alarms, results = {}, {}

    hb = heartbeat(db)
    last_co2 = last_readings(db, idx[0], now)
    collect_gap = hb is None or (now - hb).total_seconds() / 60 > COLLECT_GAP_MIN
    if collect_gap:
        current_alarms[("전체", "수집 끊김")] = f"수집기 마지막 수신 {hb.astimezone(KST):%H:%M:%S}" if hb else "수집 기록 없음"

    for r_i, r in enumerate(ROOMS):
        c = rooms_co2[r].values if r in rooms_co2 else np.full(len(idx), np.nan)
        t = rooms_tmp[r].values if r in rooms_tmp else np.full(len(idx), np.nan)
        last = last_co2.get(r, pd.NaT)
        age = (now - last.to_pydatetime()).total_seconds() / 60 if pd.notna(last) else None
        gap = age is None or age > DATA_GAP_MIN
        if gap:
            current_alarms[(r, "데이터 끊김")] = f"CO₂ 마지막 수신 {last:%H:%M}" if pd.notna(last) else "최근 2일 CO₂ 없음"
        # 마지막으로 값이 있는 칸 (끊긴 동안은 끊기기 전 값 기준, 표시만)
        ok = np.where(~np.isnan(c))[0]
        res = {"room": r, "computed_at": now.isoformat(timespec="seconds"), "model_version": VERSION,
               "data_time": idx[ok[-1]].isoformat() if len(ok) else None,
               "data_status": {"last_reading": last.isoformat() if pd.notna(last) else None,
                               "age_min": round(age, 1) if age is not None else None, "gap": bool(gap)},
               "co2_now_ppm": round(float(c[i]), 1) if not np.isnan(c[i]) else None,
               "temperature_now_c": round(float(t[i]), 2) if not np.isnan(t[i]) else None}

        # 1) 60분 예측 + 1,000ppm까지 남은 시간 + 환기 경보
        x = gru_input(c, i, idx, r_i, cfg, N_IN)
        if x is not None:
            pred = c[i] + run("gru", x[None])[0] * DS
            over = np.where(pred >= 1000)[0]
            mx30 = float(pred[:6].max())
            if c[i] >= 1000:
                level, mins = "환기 필요(이미 초과)", 0
            else:
                mins = int((over[0] + 1) * 5) if len(over) else None
                level = "환기 필요" if mx30 >= VENT_ALARM else "주의" if mx30 >= VENT_WARN else "정상"
            res["forecast"] = {"step_min": list(range(5, 5 * H + 1, 5)), "co2_ppm": [round(float(v), 1) for v in pred]}
            res["minutes_to_1000ppm"] = mins
            res["ventilation"] = {"level": level, "max_30min_ppm": round(mx30, 1)}
            if level != "정상" and level != "주의":
                current_alarms[(r, "환기")] = f"{level}: 30분 내 최대 {mx30:.0f}ppm"
        else:
            res["forecast"] = None
            res["minutes_to_1000ppm"] = None
            res["ventilation"] = {"level": "판단 불가(최근 2시간 결측)", "max_30min_ppm": None}

        # 2) 이상 탐지: 오토인코더 + 규칙
        f = ae_features(c, t, P["mc"], P["sc"], P["mt"], P["st"])
        w = ae_window(f, i)
        parts = rule_alarm(c, t, idx, parts=True)
        rules = [k for k, v in parts.items() if bool(v[i])]
        an = {"rules": rules, "score": None, "threshold": P["ae_thr"][r], "is_anomaly": bool(rules), "cause_sensor": None}
        if rules:
            an["cause_sensor"] = "temp" if ("온도 상승" in rules or (not np.isnan(t[i]) and abs(t[i] - t[i - 1]) > 2)) else "co2"
        if w is not None:
            sc, cause = ae_score(w)
            an["score"] = round(sc, 4)
            if sc > P["ae_thr"][r]:
                an["is_anomaly"], an["cause_sensor"] = True, an["cause_sensor"] or cause
                rules = rules + ["오토인코더"]
        if an["is_anomaly"]:
            current_alarms[(r, "이상")] = f"{', '.join(rules)} (원인 센서 {an['cause_sensor']})"
        res["anomaly"] = an

        # 3) 사용 여부
        F = occ_features(c, t, r_i, len(ROOMS), P["mc"], P["sc"])[i]
        if np.isnan(F).any():
            res["occupancy"] = {"prob": None, "occupied": None}
        else:
            prob = float(1 / (1 + np.exp(-run("mlp", F[None].astype(np.float32))[0])))
            res["occupancy"] = {"prob": round(prob, 3), "occupied": prob >= MLP_TH}
        results[r] = res

    events = update_alarms(now, current_alarms)
    for r, res in results.items():
        res["alarms"] = [{"종류": k[1], "설명": a["detail"], "시작": a["since"].isoformat(timespec="seconds")}
                         for k, a in active.items() if k[0] in (r, "전체")]
    return results, events


# ---------- 조회 API ----------
STATE = {"results": {}, "status": {}, "events": []}


class API(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        name = {v: k for k, v in ASCII.items()}
        if path in ("", "/"):
            body = {"endpoints": ["/health", "/rooms", "/latest", "/latest/<room>", "/alarms"],
                    "rooms": {r: ASCII.get(r, r) for r in ROOMS}}
        elif path == "/health":
            body = STATE["status"]
        elif path == "/rooms":
            body = {r: ASCII.get(r, r) for r in ROOMS}
        elif path == "/latest":
            body = STATE["results"]
        elif path.startswith("/latest/"):
            key = name.get(path.split("/")[-1], path.split("/")[-1])
            body = STATE["results"].get(key)
            if body is None:
                return self.send(404, {"error": f"없는 공간: {key}", "rooms": list(ASCII.values())})
        elif path == "/alarms":
            body = {"active": [{"공간": k[0], "종류": k[1], "설명": a["detail"], "시작": a["since"].isoformat(timespec="seconds")}
                               for k, a in active.items()], "recent_events": STATE["events"][-50:]}
        else:
            return self.send(404, {"error": "없는 주소"})
        self.send(200, body)

    def send(self, code, body):
        data = json.dumps(body, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def summary():
    """가동 로그 요약: 가동 시간, 주기 누락(2분 넘게 빈 구간), 처리 시간, 실패, 경보 내역"""
    rows = [json.loads(line) for line in open(DIR + "service_log.jsonl")]
    cyc = pd.DataFrame([r for r in rows if r["type"] == "cycle"])
    cyc["t"] = pd.to_datetime(cyc["time"])
    print(f"기록 {cyc['t'].min():%m-%d %H:%M} ~ {cyc['t'].max():%m-%d %H:%M} "
          f"({(cyc['t'].max() - cyc['t'].min()).total_seconds() / 3600:.1f}시간), 주기 {len(cyc)}회, 실패 {(~cyc['ok']).sum()}회")
    gaps = cyc["t"].diff()
    big = cyc[gaps > pd.Timedelta("2min")]
    print("주기 사이가 2분 넘게 빈 곳(서비스 중단):", "없음" if big.empty else "")
    for _, g in big.iterrows():
        print(f"  {g['t'] - gaps[g.name]:%m-%d %H:%M} → {g['t']:%H:%M} ({gaps[g.name].total_seconds() / 60:.0f}분)")
    d = cyc.loc[cyc["ok"], "ms"]
    print(f"처리 시간(ms): 평균 {d.mean():.0f}, p95 {d.quantile(0.95):.0f}, 최대 {d.max():.0f}")
    ev = pd.DataFrame([e for r in rows if r["type"] == "alarm" for e in r["events"]])
    print("\n경보 내역:" if len(ev) else "\n경보 없음")
    if len(ev):
        print(ev[[c for c in ["시각", "상태", "공간", "종류", "설명", "지속분"] if c in ev]].to_string(index=False))
    err = [r for r in rows if r["type"] == "cycle" and not r["ok"]]
    for r in err[:10]:
        print("실패:", r["time"], r.get("error", "")[:200])


def main():
    if "--summary" in args:
        return summary()
    use_mqtt = "--no-mqtt" not in args
    fixed_now = None
    if "--now" in args:
        fixed_now = pd.Timestamp(args[args.index("--now") + 1], tz="Asia/Seoul").to_pydatetime()
    db = sqlite3.connect(DIR + env.get("DB_FILE", "live.db"), timeout=30, check_same_thread=False)

    pub = None
    if use_mqtt:
        import paho.mqtt.client as mqtt
        pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        pub.username_pw_set(env["MQTT_USER"], env["MQTT_PASS"])
        pub.will_set(f"{PREFIX}/status", json.dumps({"alive": False}), qos=1, retain=True)   # 서비스가 죽으면 브로커가 알림
        pub.reconnect_delay_set(1, 30)
        pub.connect_async(env["MQTT_HOST"], int(env.get("MQTT_PORT", 1883)), keepalive=30)
        pub.loop_start()

    if "--once" not in args:
        port = int(env.get("API_PORT", 8425))
        threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), API).serve_forever, daemon=True).start()
        print(f"조회 API: http://<젯슨 주소>:{port}/latest", flush=True)

    log = open(DIR + "service_log.jsonl", "a")
    while True:
        now = fixed_now or datetime.now(KST)
        s = time.perf_counter()
        rec = {"type": "cycle", "time": now.isoformat(timespec="seconds")}
        try:
            results, events = cycle(db, now)
            ms = (time.perf_counter() - s) * 1000
            STATE["results"] = results
            STATE["events"] = (STATE["events"] + events)[-200:]
            rec.update(ok=True, ms=round(ms, 1))
            if pub:
                for r, res in results.items():
                    pub.publish(f"{PREFIX}/room/{ASCII.get(r, r)}", json.dumps(res, ensure_ascii=False), qos=1, retain=True)
                for e in events:
                    pub.publish(f"{PREFIX}/alarm", json.dumps(e, ensure_ascii=False), qos=1, retain=False)
            if events:
                log.write(json.dumps({"type": "alarm", "time": rec["time"], "events": events}, ensure_ascii=False) + "\n")
                for e in events:
                    print(f"[경보 {e['상태']}] {e['공간']} {e['종류']} {e.get('설명', '')}", flush=True)
        except Exception as e:
            rec.update(ok=False, ms=round((time.perf_counter() - s) * 1000, 1), error=f"{type(e).__name__}: {e}")
            print(f"{now:%F %T} 주기 실패: {rec['error']}\n{traceback.format_exc()}", flush=True)
        STATE["status"] = {"alive": True, "last_cycle": rec["time"], "ok": rec["ok"], "ms": rec["ms"], "version": VERSION,
                           "active_alarms": len(active)}
        if pub:
            pub.publish(f"{PREFIX}/status", json.dumps(STATE["status"], ensure_ascii=False), qos=1, retain=True)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        if "--once" in args:
            print(json.dumps(STATE["results"], ensure_ascii=False, indent=1, default=str))
            break
        time.sleep(60 - time.time() % 60 + 1)          # 다음 분 정각 + 1초


if __name__ == "__main__":
    main()
