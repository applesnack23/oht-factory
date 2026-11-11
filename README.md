# AI OHT Fab — 반도체 OHT 가상 팹 + AI 배차

중간 규모 반도체 팹의 OHT(Overhead Hoist Transport) 반송 시스템을 시뮬레이션하는 3D 디지털 트윈입니다. 강화학습/모방학습 AI가 반송 결정을 내립니다. `ai-conveyor-factory`의 자매 프로젝트이고, 포트가 달라 두 프로젝트를 동시에 띄울 수 있습니다.

| 항목 | 값 |
|---|---|
| 팹 크기 | 약 130 m × 110 m, 레일 높이 4.5 m |
| 레일 | 단방향 인터베이 링 + 지름길 2개(XD, XU) + 베이 8개(인트라베이 루프) + 스토커 2개 |
| 포트 | 120개 (베이당 14개 + 스토커 8개) |
| 차량 | 50대 (fleet_short 시나리오는 36대) |
| 수요 | 분당 16건 기본, 공정 흐름(LITHO→ETCH→DEPO→…), 핫랏(마감 300 s) |

## AI가 내리는 결정 3가지

| 결정 | 언제 | 선택지 |
|---|---|---|
| **DISPATCH (배차)** | 새 반송 작업이 대기 중일 때 | 후보 차량 최대 8대 중 1대, 또는 "보류" |
| **ROUTE (경로)** | 차량이 지름길 분기점에 다가갈 때 | 최대 8개 경로 후보 중 1개 (지름길 / 링 유지) |
| **REPOSITION (재배치)** | 차량이 할 일 없이 놀고 있을 때 | 최대 8개 대기 위치(베이/스토커) 중 1곳, 또는 그대로 |

한 step이 결정 하나입니다. 관측은 111차원(전역 16 + 결정 종류 3 + 맥락 12 + 후보 8×10), 행동은 `MultiDiscrete([9])`(후보 8개 + 기본 행동)입니다. 쓸 수 없는 후보는 action mask로 막습니다.

## 실행 (Mac)

### 1) 파이썬 환경
컨베이어 프로젝트의 `.venv`를 그대로 써도 되고, 새로 만들어도 됩니다.

```bash
cd ~/project/digital-factory/ai-oht-fab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

컨베이어 venv를 재사용하려면 `source ../ai-conveyor-factory/.venv/bin/activate`를 실행하세요.

### 2) 백엔드 (터미널 1)
```bash
source .venv/bin/activate
python -m backend.main          # http://localhost:8001  (FastAPI + /ws/fab)
```
FastAPI가 설치되어 있지 않으면 자동으로 표준 라이브러리 서버로 동작합니다. 같은 포트, 같은 API를 씁니다.

### 3) 프론트엔드 (터미널 2)
```bash
cd frontend
npm install                     # 처음 한 번
npm run dev                     # http://localhost:5174
```
브라우저에서 **http://localhost:5174** 를 엽니다. 컨베이어 프로젝트(8000/5173)와 동시에 실행해도 충돌하지 않습니다.

### 4) 화면 사용법
- 위쪽 **Start / Pause / Reset** 은 시뮬레이션 재생 버튼입니다. Speed는 x1–x50, Scenario로 7개 상황을 고릅니다.
- **Control**:

  | 모드 | 동작 |
  |---|---|
  | `rule_basic` | 가장 가까운 빈 차량 배차(NVF) |
  | `rule_smart` | 핫랏 우선, 정체를 고려한 경로, 수요 기반 재배치 |
  | `ppo` | 아래 Model에서 고른 AI 모델 |

- **Model**: `oht_bc`(모방학습), `oht_ppo_a`, `oht_ppo_b`(PPO 미세조정) 중 하나를 고르고 **Load**를 누릅니다.
- 오른쪽 패널에는 KPI, 베이별 대기, 최근 AI 결정 로그(어떤 결정을 왜 했는지)가 나옵니다.
- 3D 범례:

  | 표시 | 의미 |
  |---|---|
  | 차량 색 | 상태 |
  | 흰 상자 | FOUP |
  | 빨간 점 | 핫랏 |
  | 보라색 레일 | 지름길 (빨간색이면 폐쇄) |
  | 분홍 점 | 경로 결정 지점 |

- **Training** 패널에서 시나리오(`mix` 권장)와 스텝 수를 정하고 Start training을 누르면 학습이 시작됩니다. **Watch** 를 누르면 학습 중인 모델의 최신 스냅샷이 3D로 실행되고, 새 스냅샷이 나올 때마다 자동으로 다시 로드됩니다.

## 시나리오

| 이름 | 내용 |
|---|---|
| normal | 기본 수요 16건/분, 차량 50대 |
| high_load | 수요 +40% (22.4건/분) |
| hot_lots | 핫랏 20% |
| bay_surge | 300–1500 s 동안 LITHO/ETCH 수요 급증 |
| fleet_short | 차량 36대만 운행 |
| track_block | 지름길 XD를 300–1200 s 동안 폐쇄 |
| mixed | 고부하 + 핫랏 12% + XU 600–1300 s 폐쇄 |

## 학습 / 평가 명령

```bash
# 1. 모방학습(BC + DAgger): smart 규칙을 흉내 내서 시작점을 만듦 (약 5분)
python -m backend.ai.pretrain_bc --level mix --run oht_bc

# 2. PPO 미세조정 (BC에서 시작)
python -m backend.ai.train_ppo --level mix --run my_ppo --init-from oht_bc \
    --hidden 128 128 --lr 1e-4 --clip-range 0.1 --ent-coef 0.001 --n-epochs 5 --timesteps 600000

# 처음부터 PPO (비교용, 성능 낮음)
python -m backend.ai.train_ppo --level normal --run scratch --timesteps 300000

# 동작 확인용 초소형 학습
python -m backend.ai.train_ppo --level normal --smoke

# 3. 비교 평가 (7개 시나리오 × 5 seed, 학습에 쓰지 않은 seed)
python -m backend.ai.evaluate --policies random rule_basic rule_smart ppo:oht_bc ppo:oht_ppo_a --seeds 5 --name compare
#   → backend/logs/eval/compare/report.md, comparison.png

# TensorBoard (SB3로 학습했을 때)
tensorboard --logdir backend/logs
```

Mac(.venv에 SB3/torch 있음)에서는 SB3 PPO가 자동으로 쓰입니다(`--algo auto`). torch가 없으면 내장 NumpyPPO로 학습합니다. 두 방식의 모델 모두 3D 화면에서 실행할 수 있습니다.

## 테스트
```bash
python -m pytest -q backend/tests      # 시뮬레이션 안전성, 환경, 규칙, 학습/BC, 서버/WebSocket
cd frontend && node --test tests/
```

## 결과
[PROGRESS.md](PROGRESS.md)에 평가 표가 있습니다. 결과는 과장 없이 적었습니다.
