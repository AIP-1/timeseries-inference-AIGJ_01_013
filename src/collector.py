import json
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import paho.mqtt.client as mqtt

# 과제 8 — 수집기: MQTT 센서 업링크 → 젯슨 로컬 SQLite (상시 서비스 service.py의 입력)
# 접속 정보는 코드에 쓰지 않고 ~/ts-data/.env 에서 읽음 (git에 올리지 않기)
# 저장: readings(time UTC ISO, device_id, location, meas, value), 같은 업링크 중복(deduplicationId+항목)은 한 번만
#       heartbeat(마지막 수신 시각) — 서비스가 "수집 끊김"을 판단할 때 씀
# 7일 넘은 기록은 1시간마다 지움
DIR = os.path.expanduser("~/ts-data/")
MEAS = {"co2": "co2", "temperature": "temperature", "humidity": "humidity"}   # 업링크 object 키 → 저장 이름
KEEP_DAYS = 7


def load_env(path=DIR + ".env"):
    env = {}
    if not os.path.exists(path):          # .env 없이 --no-mqtt 시험할 때
        return env
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    return env


def open_db(path):
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")       # 서비스가 읽는 동안에도 쓸 수 있게
    db.execute("""CREATE TABLE IF NOT EXISTS readings (time TEXT, device_id TEXT, location TEXT, meas TEXT, value REAL,
                  dedup TEXT, UNIQUE(dedup, meas))""")
    db.execute("CREATE INDEX IF NOT EXISTS ix_time ON readings(time)")
    db.execute("CREATE TABLE IF NOT EXISTS heartbeat (id INTEGER PRIMARY KEY, last_rx TEXT, n_total INTEGER)")
    db.commit()
    return db


def to_utc(t):
    """ISO 문자열 → UTC datetime. 소수 초가 9자리(나노초)여도 받도록 6자리로 자름"""
    t = re.sub(r"(\.\d{6})\d+", r"\1", t.replace("Z", "+00:00"))
    return datetime.fromisoformat(t).astimezone(timezone.utc)


def parse(payload):
    """ChirpStack 업링크 JSON → [(time, device_id, location, meas, value, dedup)]. 해당 항목이 없으면 빈 목록"""
    m = json.loads(payload)
    obj, info = m.get("object") or {}, m.get("deviceInfo") or {}
    loc = (info.get("tags") or {}).get("location")
    dev = info.get("deviceName") or info.get("devEui")
    # 시각: 네트워크 서버가 받은 시각(rxInfo.nsTime, 가장 이른 것). "time"은 게이트웨이 시계라 틀린 게이트웨이가 있음
    # (10/21 확인: 한 게이트웨이가 약 5분 빠름). nsTime이 없으면 time 사용
    ns = [r["nsTime"] for r in (m.get("rxInfo") or []) if r.get("nsTime")]
    t = min(ns, key=to_utc) if ns else m.get("time")
    if not (loc and dev and t):
        return []
    ts = to_utc(t).isoformat()
    dedup = m.get("deduplicationId") or f"{dev}-{t}"
    return [(ts, dev, loc, name, float(obj[k]), dedup) for k, name in MEAS.items()
            if isinstance(obj.get(k), (int, float))]


def main():
    env = load_env()
    db = open_db(DIR + env.get("DB_FILE", "live.db"))
    state = {"n": 0, "last_clean": 0}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"{datetime.now():%F %T} 브로커 연결 ({reason_code}), 구독 {env['SUB_TOPIC']}", flush=True)
        client.subscribe(env["SUB_TOPIC"], qos=0)

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        print(f"{datetime.now():%F %T} 연결 끊김 ({reason_code}) — 자동 재연결", flush=True)

    def on_message(client, userdata, msg):
        try:
            rows = parse(msg.payload)
        except Exception as e:                    # 형식이 다른 메시지는 건너뜀 (서비스는 계속)
            print(f"{datetime.now():%F %T} 해석 실패 {msg.topic}: {e}", flush=True)
            return
        now = datetime.now(timezone.utc).isoformat()
        if rows:
            db.executemany("INSERT OR IGNORE INTO readings VALUES (?,?,?,?,?,?)", rows)
            state["n"] += len(rows)
        db.execute("INSERT OR REPLACE INTO heartbeat VALUES (1, ?, ?)", (now, state["n"]))
        db.commit()
        if time.time() - state["last_clean"] > 3600:
            cut = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).isoformat()
            db.execute("DELETE FROM readings WHERE time < ?", (cut,))
            db.commit()
            state["last_clean"] = time.time()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=env.get("CLIENT_ID", "") or "")
    client.username_pw_set(env["MQTT_USER"], env["MQTT_PASS"])
    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)      # 브로커가 재시작되면 30초 안에 다시 붙음
    while True:
        try:
            client.connect(env["MQTT_HOST"], int(env.get("MQTT_PORT", 1883)), keepalive=30)
            client.loop_forever(retry_first_connection=True)
        except Exception as e:
            print(f"{datetime.now():%F %T} 접속 실패: {e} — 5초 뒤 재시도", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
