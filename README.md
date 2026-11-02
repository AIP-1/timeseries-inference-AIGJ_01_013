# timeseries-inference — 실내 환경 시계열 추론

AIGJ_01_013 이진행

실습실·사무실·회의실 3곳의 12주 실측 CO₂·온도 데이터로 예측·이상 탐지·상태 추정 모델을 만들고, Jetson Orin Nano Super에서 매분 도는 상시 추론 서비스로 가동한 과제입니다.

- **최종 보고서:** [report/최종보고서.md](report/%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5.md) ([PDF](report/%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5.pdf))
- **결과 형식 문서:** [docs/과제8_결과형식.md](docs/%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A68_%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%E1%84%92%E1%85%A7%E1%86%BC%E1%84%89%E1%85%B5%E1%86%A8.md)

## 핵심 결과

| 과제 | 결과 |
|---|---|
| 2. 정제·기준선 | 규칙 7개로 5분 간격 정제, 시간순 분할(학습 7/27~9/20, 검증 9/21~10/4, 시험 10/5~10/18). 직전값 유지가 가장 강한 기준선 |
| 3. GRU 예측 | 60분 CO₂ 오차를 직전값보다 10~22% 줄임(목표 30% 미달), 온도는 전 공간 개선 |
| 4. 이상 탐지 | README 고장 모양 30건 주입, 결합 탐지 83%, 헛경보 1.14회/일 |
| 5. 상태 추정 | 사용 여부 MLP F1 67.2(규칙 49.9), 환기 경보는 GRU 30분 예측 ≥ 860ppm으로 재현율 89.6% |
| 6. 사전학습 모델 | Chronos-2가 학습 없이 GRU보다 정확(60분 MAE 34.2/19.5/56.6 vs 36.7/25.2/60.4ppm) |
| 7. 엣지 최적화 | ONNX·TensorRT 변환 차이 기준 통과(최대 0.21ppm), 작은 모델은 ONNX Runtime CPU가 가장 빠름(0.03~0.19ms) |
| 8. 상시 서비스 | systemd로 17.8시간 1,068주기 실패 0회, 평균 985ms, 수집기 정지 감지·해제 확인 |

가장 의외였던 발견: 9/21부터 세 공간 CO₂가 한꺼번에 크게 낮아져, 학습 기간 패턴을 기억하는 모델은 모두 흔들렸고 매번 7일치를 다시 읽는 Chronos-2만 강했습니다.

## 폴더 구성

```
report/        최종 보고서 (Markdown, PDF, 그림)
docs/          과제별 결과 기록, 정제 규칙, 상시 서비스 설치·결과 형식
src/           최종 코드
  service/     systemd 유닛과 .env 견본
  archive/     실험 중간 버전 (anomaly_v1~v11 등, 보고서의 버전 비교 근거)
```

## 코드와 실행 순서

모든 스크립트는 젯슨의 `~/ts-data/`에서 실행했습니다. 실행 전 가상환경을 켭니다: `source ~/ts-venv/bin/activate`

| 순서 | 파일 | 과제 | 하는 일 |
|---|---|---|---|
| 1 | `clean.py` | 2 | 원본(pkl)을 정제해 `{co2,temperature}_room_5min.pkl` 생성 |
| 2 | `baseline.py` | 2 | 기준선 3종 오차, 평가 시점 `eval_points.csv` 저장 |
| 3 | `train_gru.py` | 3 | GRU 60분 예측 학습·평가 |
| 4 | `anomaly.py` | 4 | 고장 주입, 오토인코더+규칙 이상 탐지 (최종 v10) |
| 5 | `state.py`, `vent_gru.py` | 5 | 사용 여부 MLP, GRU 예측 기반 환기 경보 |
| 6 | `pretrained.py` | 6 | Chronos-2 / Chronos-Bolt 무학습 예측 비교 |
| 7 | `edge.py` | 7 | ONNX·TensorRT 변환, 차이와 지연 측정 |
| 8 | `features.py`, `collector.py`, `service.py` | 8 | 학습·서비스 공용 전처리, MQTT 수집기, 매분 추론 서비스 |

상시 서비스 설치 방법은 [docs/과제8_설치.md](docs/%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A68_%E1%84%89%E1%85%A5%E1%86%AF%E1%84%8E%E1%85%B5.md)에 있습니다.

## 환경

- Jetson Orin Nano Super 8GB, JetPack 7.2, 전력 모드 MAXN_SUPER
- Python 3.12, torch 2.14.1+cu132, onnxruntime 1.31.0, TensorRT 10.16.2.10, numpy 1.26.4

## 올리지 않은 것

- 센서 원본·정제 데이터와 학습된 모델 파일(용량, 데이터 공개 범위)
- 접속 정보(`.env`): MQTT·InfluxDB 주소와 계정은 `src/service/env.example` 형식으로 각자 채웁니다.
