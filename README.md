# AI OHT Fab — 반도체 OHT 가상 팹 + AI 배차

중간 규모 반도체 팹의 OHT(Overhead Hoist Transport) 반송 시스템을 시뮬레이션하는 3D 디지털 트윈입니다. 강화학습/모방학습 AI가 반송 결정을 내립니다. `ai-conveyor-factory`의 자매 프로젝트이고, 포트가 달라 두 프로젝트를 동시에 띄울 수 있습니다.

| 항목 | 값 |
|---|---|
| 팹 크기 | 약 130 m × 110 m, 레일 높이 4.5 m |
| 레일 | 단방향 인터베이 링 + 지름길 2개(XD, XU) + 베이 8개(인트라베이 루프) + 스토커 2개 |
| 장비 | 64대 (베이당 7–8대). 장비마다 EFEM과 로드포트 1–2개, 로드포트 112개 + 스토커 포트 8개 |
| 차량 | 50대 (fleet_short 시나리오는 36대) |
| 물류 흐름 | 닫힌 루프: 랏 약 85개가 공정 흐름(LITHO→ETCH→DEPO→…)을 따라 장비와 스토커를 계속 오감. 반송 요청은 공정이 끝난 FOUP에서 생김. 핫랏 5% (반송 마감 300 s) |

## 팹 동작 (닫힌 루프)

1. 장비의 로드포트에 FOUP가 놓이면 EFEM 쪽으로 밀려 붙고(도킹) 문이 열립니다. FOUP는 장비 안으로 들어가지 않고, 웨이퍼만 EFEM 로봇이 장비로 옮긴다고 가정합니다.
2. 장비는 랏을 한 번에 하나씩 처리합니다(공정별 2–5분, ±25%). 다른 로드포트에 도킹된 랏은 차례를 기다립니다.
3. 공정이 끝나면 FOUP가 도킹에서 풀리고, 다음 공정으로 가는 반송 요청이 생깁니다.
4. 목적지는 다음 공정 장비의 **빈 로드포트**이고, 즉시 예약됩니다. 그래서 FOUP가 있는 로드포트에 또 내려놓는 일이 없습니다. 빈 로드포트가 없거나 대기해야 하면 스토커로 갑니다.
5. 스토커에 있는 랏은 해당 공정 장비의 로드포트가 비면 그쪽으로 보내집니다.

반송이 빠르면 장비가 덜 기다리므로 장비 가동률과 처리량이 올라갑니다. 배차 품질이 팹 생산성으로 이어지는 구조입니다. 랏마다 다음 공정과 처리 시간은 (seed, 랏, 단계)로 정해지므로, 어떤 정책으로 돌려도 같은 랏을 상대합니다.

## AI가 내리는 결정 3가지

| 결정 | 언제 | 선택지 |
|---|---|---|
| **DISPATCH (배차)** | 새 반송 작업이 대기 중일 때 | 후보 차량 최대 8대 중 1대, 또는 "보류". 후보에는 놀고 있는 차량과 **곧 짐을 내릴 차량(선배정)**이 함께 들어감 |
| **ROUTE (경로)** | 차량이 지름길 분기점에 다가갈 때 | 최대 8개 경로 후보 중 1개 (지름길 / 링 유지) |
| **REPOSITION (재배치)** | 차량이 할 일 없이 놀고 있을 때 | 최대 8개 대기 위치(베이/스토커) 중 1곳, 또는 그대로 |

**선배정(pre-assignment)**: 짐을 내리러 가는 중인 차량에게 다음 작업을 미리 예약해 두면, 그 차량은 짐을 내리자마자 바로 다음 작업으로 출발합니다. 실제 OHT 제어 시스템(MCS)에서도 쓰는 방식입니다. 이 기능을 `config/fab.json`의 `dispatch.preassign`으로 켜고 끕니다.

한 step이 결정 하나입니다. 관측은 127차원(전역 16 + 결정 종류 3 + 맥락 12 + 후보 8×12), 행동은 `MultiDiscrete([9])`(후보 8개 + 기본 행동)입니다. 쓸 수 없는 후보는 action mask로 막습니다.

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
  | `rule_preassign` | smart + 곧 짐을 내릴 차량에게 선배정 (AI의 선생님 규칙) |
  | `ppo` | 아래 Model에서 고른 AI 모델 |

- **Model**: `oht_ppo3_a`, `oht_ppo3_b`, `oht_bc3` 중 하나를 고르고 **Load**를 누릅니다. 최근 평가(v3.3)에서는 `oht_ppo3_a`가 AI 중 가장 좋았고, 선배정 규칙(`rule_preassign`)과 비슷했습니다(PROGRESS.md 참고).
- 오른쪽 패널에는 KPI, 베이별 대기, 최근 AI 결정 로그(어떤 결정을 왜 했는지)가 나옵니다.
- 3D 범례:

  | 표시 | 의미 |
  |---|---|
  | 차량 색 | 상태 |
  | 흰 상자 | FOUP |
  | 빨간 점 | 핫랏 |
  | 보라색 레일 | 지름길 (빨간색이면 폐쇄) |
  | 분홍 점 | 경로 결정 지점 |

- 화면 오른쪽 위 시점 버튼:

  | 버튼 | 보이는 것 |
  |---|---|
  | **Stocker 1 / Stocker 2** | 스토커 내부. 스태커 크레인(노란 기둥)이 포트와 선반 사이에서 FOUP를 옮김 |
  | **Hoist** | 지금 FOUP를 올리거나 내리는 OHT를 가까이에서 봄 |
  | **Tool** | FOUP가 도킹·언도킹되는 로드포트(없으면 공정 중인 로드포트)를 가까이에서 봄 |

  OHT는 로드포트 위에 서면 벨트 4줄로 그리퍼를 내리고, FOUP를 집거나 놓은 뒤 다시 올립니다. 호이스트 시간은 12초입니다.
- 장비: 뒤쪽 큰 상자가 공정 모듈(공정 색), 앞쪽 밝은 회색이 EFEM, 그 앞의 받침대가 로드포트입니다. EFEM 지붕 끝에 **타워램프**가 있습니다(위에서부터 빨강·주황·초록·파랑). 여러 칸이 함께 켜질 수 있습니다.

  | 타워램프 | 의미 |
  |---|---|
  | 빨강 | 알람(깜빡임) / 매뉴얼 모드(켜짐). 시뮬레이션에서는 생기지 않음 |
  | 주황 | 공정 중이 아님 |
  | 초록 | 공정 완료, OHT에 반송 요청함 (가져가길 기다림) |
  | 파랑 | 공정 중 |

  예를 들어 파랑 + 초록이 함께 켜져 있으면, 한 로드포트는 공정 중이고 다른 로드포트의 랏은 OHT를 기다리는 상태입니다.
  빨강은 API로 확인할 수 있습니다. `POST /api/fab/tool-mode {"tool": "B2-T02", "mode": "alarm" | "manual" | "auto"}` 알람이나 매뉴얼 상태인 장비는 새 랏을 시작하지 않습니다.

  로드포트 판의 작은 램프와 막대는 로드포트별 상태입니다.

  | 로드포트 램프 | 의미 |
  |---|---|
  | 파랑(깜빡임) | 공정 중. 파란 막대가 진행률 |
  | 초록 | 공정 끝, OHT 대기 |
  | 하늘색 | 도킹 / 언도킹 중 |
  | 주황 | 도킹됨, 장비 차례 대기 |
  | 흰색 | 이 로드포트로 FOUP가 오는 중(예약) |
  | 회색 | 비어 있음 |
- 카메라 조작:

  | 동작 | 결과 |
  |---|---|
  | 트랙패드 두 손가락 쓸기 | 시점 이동(바닥을 따라 화면이 움직임) |
  | 트랙패드 핀치(벌리기/오므리기) | 확대 / 축소 |
  | 마우스 휠 | 확대 / 축소 (Shift + 휠 = 옆으로 이동) |
  | 왼쪽 드래그 / 오른쪽 드래그 | 회전 / 이동 |
- 스토커 포트는 입력(IN, 초록 벨트) 2개와 출력(OUT, 파란 벨트) 2개로 나뉩니다. 스토커로 가는 FOUP는 IN 포트에 내려놓고, 스토커에서 나오는 FOUP는 OUT 포트에서 가져갑니다. 레일 진행 방향으로 IN이 먼저, OUT이 뒤에 있습니다. 포트마다 스테이지(레일 아래)와 안쪽 끝, 두 자리가 있고 한 자리에 FOUP는 하나만 놓입니다.
  - 입고: OHT가 IN 스테이지에 FOUP를 놓으면, 안쪽 끝이 비는 대로 컨베이어가 실어 나르고(2.5초) 크레인이 집어 선반에 넣습니다. IN 스테이지가 차 있으면 OHT는 그리퍼를 내리지 않고 기다립니다.
  - 출고: 크레인은 OUT 포트 안쪽 끝이 비어 있을 때만 FOUP를 내려놓고, 스테이지가 비면 컨베이어가 내보냅니다.
  - OUT 포트에 온 OHT는 스테이지에 있는 FOUP를 가져갑니다. 그 FOUP가 같은 포트의 다른 작업 것이면 두 작업의 담당 OHT를 서로 바꿉니다(MCS 재바인딩). 그래서 내 FOUP가 다른 FOUP 뒤에 막혀 서로 기다리는 일이 없습니다.
  - 선반과 컨베이어는 FOUP를 양쪽 두 받침으로 받치고 가운데에 홈이 있습니다. 크레인 포크는 FOUP보다 낮게 홈으로 들어가서, 천천히 올라가 FOUP를 떠 올린 뒤 빠져나옵니다. 내려놓을 때는 높게 들어가 FOUP가 받침에 닿을 때까지 내려온 뒤 빠져나옵니다.
  - 포크는 턴테이블 위에 있어서, 반대편 선반(통로 건너편)에 넣거나 꺼낼 때는 180° 돌아서 들어갑니다(1.2초).
  - OHT가 FOUP보다 먼저 스테이지에 도착하면 그리퍼를 올린 채 기다립니다.
- FOUP 윗면에는 손잡이(플랜지)가 있고, OHT 그리퍼의 두 집게 발이 내려와 플랜지 아래로 오므려 잡습니다. 내려놓은 뒤에는 벌린 채로 올라갑니다.
- **Training** 패널에서 시나리오(`mix` 권장)와 스텝 수를 정하고 Start training을 누르면 학습이 시작됩니다. **Watch** 를 누르면 학습 중인 모델의 최신 스냅샷이 3D로 실행되고, 새 스냅샷이 나올 때마다 자동으로 다시 로드됩니다.

## 시나리오

| 이름 | 내용 |
|---|---|
| normal | 기본 팹 (장비 64대, WIP 약 70%, 차량 50대) |
| high_load | 장비가 40% 빨라짐: 시간당 끝나는 랏과 반송이 늘어남 |
| hot_lots | 핫랏 20% |
| bay_surge | 300–1500 s 동안 LITHO 2.5배·ETCH 2배 빨라져 이 베이에서 랏이 몰려 나옴 |
| fleet_short | 차량 36대만 운행 |
| track_block | 지름길 XD를 300–1200 s 동안 폐쇄 |
| mixed | 장비 25% 빨라짐 + 핫랏 12% + XU 600–1300 s 폐쇄 |

## 학습 / 평가 명령

```bash
# 1. 모방학습(BC + DAgger): smart 규칙을 흉내 내서 시작점을 만듦 (약 5분)
python -m backend.ai.pretrain_bc --level mix --run oht_bc3          # 선생님: rule_preassign (--teacher smart 로 변경 가능)

# 2. PPO 미세조정 (BC에서 시작)
python -m backend.ai.train_ppo --level mix --run my_ppo --init-from oht_bc3 \
    --hidden 128 128 --lr 1e-4 --clip-range 0.1 --ent-coef 0.001 --n-epochs 5 --timesteps 600000

# 처음부터 PPO (비교용, 성능 낮음)
python -m backend.ai.train_ppo --level normal --run scratch --timesteps 300000

# 동작 확인용 초소형 학습
python -m backend.ai.train_ppo --level normal --smoke

# 3. 비교 평가 (7개 시나리오 × 5 seed, 학습에 쓰지 않은 seed)
python -m backend.ai.evaluate --policies random rule_basic rule_smart rule_preassign ppo:oht_bc3 ppo:oht_ppo3_a --seeds 5 --name compare
#   → backend/logs/eval/compare/report.md, comparison.png

# TensorBoard (SB3로 학습했을 때)
tensorboard --logdir backend/logs
```

Mac(.venv에 SB3/torch 있음)에서는 SB3 PPO가 자동으로 쓰입니다(`--algo auto`). torch가 없으면 내장 NumpyPPO로 학습합니다. 두 방식의 모델 모두 3D 화면에서 실행할 수 있습니다.

## 테스트
```bash
python -m pytest -q backend/tests      # 시뮬레이션 안전성, 환경, 규칙, 학습/BC, 서버/WebSocket
cd frontend && npm test
```

## 결과
[PROGRESS.md](PROGRESS.md)에 평가 표가 있습니다. 결과는 과장 없이 적었습니다.
