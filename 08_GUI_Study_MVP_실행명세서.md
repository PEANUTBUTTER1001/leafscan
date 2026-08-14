# LeafScan Lab GUI Study 실행 모드 — MVP 실행 명세서

> 기준: `06_GUI_Study_실행모드_SRS.md`, MVP 명세 세션 `spec-1`

## 1. 프로젝트 개요

기존 Python/Tkinter LeafScan Lab에서 작물·생육 단계 멀티헤드 이미지 분류 학습을 실행한다. 산출물은 로컬 데스크톱 GUI의 학습 실행 확장으로, 사용자는 좌측 사이드바 최상단에서 단일 실행 또는 Study 실행을 선택한다. Study 실행은 기존 `run_study.py`를 호출해 선택 모델을 순차 비교한다.

## 2. 문제 정의와 목표 지표

| 구분 | 명세 |
|---|---|
| ML 입력 | `data/index.csv`가 가리키는 이미지와 기존 전처리 설정 |
| ML 출력 | `crop`, `stage` head의 분류 logits 및 기존 학습 지표 |
| GUI 입력 | 실행 모드, 공통 학습 설정, Study 이름, 비교 모델 목록, LR 탐색 여부 |
| GUI 출력 | 단일 run 또는 Study 폴더, 로그, 큐 상태, `STUDY.md` |
| 기능 성공 기준 | 유효한 Study 요청이 순차 실행되고 Study 산출물이 생성된다. |
| 모델 지표 | 기존 프로젝트의 `stage macro-F1`, 정식기 recall, crop accuracy를 그대로 비교한다. 특정 수치 목표는 이번 GUI 확장 범위에서 새로 정하지 않는다. |

## 3. 데이터 명세

| 항목 | 명세 |
|---|---|
| 출처 | 기존 `data/index.csv` 및 로컬 이미지 파일 |
| 핵심 컬럼 | `path`, `crop`, `stage`, `group_id`, `split_source` 등 기존 index CSV 스키마 |
| 라벨 | `crop`, `stage` 멀티헤드 분류 라벨 |
| 분할 | 단일 실행은 기존 설정을 따르고, Study는 `runs/<study>/split.json`을 생성 또는 재사용 |
| 변경 여부 | 데이터 스키마·추출·전처리 규칙은 변경하지 않음 |

## 4. 실행 파이프라인

```text
GUI 공통 설정 수집
  ├─ 단일 실행: 기존 config 생성 → train_worker.py → runs/<run>/
  └─ Study 실행: Study base config 생성 → run_study.py
        → split.json 생성/재사용
        → 선택 모델별 train_worker.py 순차 실행
        → metrics.json / best.pt 저장
        → STUDY.md 생성
```

## 5. UI와 실행 방법

### 5.1 좌측 사이드바 구성

기존 사이드바의 최상단에 아래 컨트롤만 추가한다.

```text
실행 모드:  (●) 단일 실행   ( ) Study 실행
```

기존 데이터셋·공통 학습 설정 카드의 순서와 스타일은 유지한다.

### 5.2 모드별 제어 규칙

| 항목 | 단일 실행 | Study 실행 |
|---|---|---|
| `arch` 콤보박스 | 활성 | 비활성 — 비교 모델 체크 목록이 대체 |
| `simple_cnn` 전용 구조값 | 활성 | 비활성 — 특정 모델 전용 설정 |
| 가설 입력 | 활성 | 비활성 또는 숨김 |
| Study 이름 | 숨김 | 활성, 기본 `study_01_backbone` |
| 비교 모델 체크 목록 | 숨김 | 활성, 최소 2개 |
| LR 자동 탐색 | 숨김 | 활성, 선택 시 `--lr-search` |
| 데이터/전처리/epoch/batch/optimizer 등 | 활성 | 활성, 모든 선택 모델에 공통 적용 |
| 시작 버튼 | 기존 단일 실행 문구 | `Study 큐 시작` |

비활성 항목에는 아래 설명을 제공한다.

- arch: `Study 모드에서는 비교 모델 목록으로 모델을 선택합니다.`
- simple_cnn 전용 옵션: `특정 모델 전용 설정은 공정 비교에 사용하지 않습니다.`
- 가설 입력: `Study 이름과 결과표가 실험 묶음을 식별합니다.`

### 5.3 Study 입력 검증

| 조건 | 처리 |
|---|---|
| Study 이름이 비어 있음 | 시작하지 않고 이름 입력 안내 |
| 선택 모델 0개 또는 1개 | 시작하지 않고 `Study 실행에는 모델을 2개 이상 선택하세요.` 표시 |
| 2개 이상 선택 | 선택된 순서로 `--vary arch=a,b,...` 구성 |
| LR 탐색 선택 | `--lr-search` 인자 추가 |

## 6. 모델과 비교 방법

| 대상 | 처리 |
|---|---|
| 모델 목록 | `tasks.models_registry.list_archs()` 또는 기존 `ARCHS`를 기준으로 표시 |
| 최대 선택 수 | 등록된 현재 모델 수까지. 현재는 6개 |
| 순서 | GUI에 표시된 선택 모델 순서 |
| 사전학습/해상도/freeze/epoch/batch | 모든 선택 모델에 GUI 공통값을 적용 |
| 학습률 | 공통 LR 또는 모델별 LR 탐색 결과 |
| 모델 생성/학습 | `run_study.py`와 `train_worker.py`의 기존 경로 재사용 |

## 7. 평가와 재현성

- Study는 기존 `run_study.py`의 `ensure_split()`을 통해 `split.json`을 1회 만들고 모든 모델이 공유한다.
- 시드·데이터셋·전처리·공통 학습 설정은 Study base config에 저장한다.
- 결과 비교는 `metrics.json`과 자동 생성된 `STUDY.md`를 기준으로 한다.
- 모델별 기본 평가지표는 stage macro-F1, primary recall, crop accuracy, 파라미터 수, 소요 시간이다.
- 데이터 누수 방지는 기존 `group_id` 기반 분할 검증과 `split.json` 공유 규칙을 변경하지 않아 보장한다.

## 8. 산출물과 사용 방법

| 실행 | 저장 위치 | 주요 산출물 |
|---|---|---|
| 단일 실행 | `runs/<run-id>/` | 기존 `config.json`, `metrics.json`, `best.pt`, 로그/리포트 |
| Study 실행 | `runs/<study-name>/` | `split.json`, 모델별 run 폴더, `STUDY.md` |

Study 실행 중 우측 기존 로그 영역을 사용하며, 큐 상태에는 각 모델의 `대기`, `실행 중`, `완료`, `실패`, `건너뜀`을 표시한다. 기존 중단 버튼은 Study의 상위 프로세스도 종료해야 한다. 같은 Study 이름으로 재실행하면 기존 `metrics.json`이 있는 모델은 건너뛴다.

## 9. 실제 수정 범위

```text
leafscan/
├─ lab_gui.py                 # 주 수정 대상: 모드 상태, 조건부 UI, Study 프로세스/로그/중단 관리
├─ run_study.py               # 수정하지 않고 재사용
├─ train_worker.py            # 수정하지 않고 재사용
├─ core/
│  ├─ data.py                 # 기존 split/데이터 처리 재사용
│  └─ split.py                # 기존 group 기반 분할 재사용
├─ tasks/models_registry.py   # 기존 모델 목록 재사용
└─ runs/<study-name>/         # Study 실행 산출물
```

## 10. 개발 Phase

| Phase | 목표 | 작업 | 산출물 |
|---|---|---|---|
| 1 | 상태와 UI 전환 | 실행 모드·Study 이름·선택 모델·LR 탐색 상태 변수 추가, 모드 선택 컨트롤과 Study 프레임 추가 | 단일/Study UI 전환 가능 |
| 2 | 단일 모드 보존 | 기존 config 수집과 `start_training()` 경로를 단일 모드로 고정, Study 모드의 단일 모델 전용 컨트롤 비활성화 | 단일 실행 회귀 없음 |
| 3 | Study 명령 연결 | GUI 공통 설정으로 base config 생성, `run_study.py --study --base --vary` 프로세스 실행 | 선택 모델 순차 실행 |
| 4 | 관찰·중단·재개 | 표준 출력을 GUI 로그에 연결, 모델 상태 파싱/표시, 상위 프로세스 중단 지원 | 큐 상태와 중단 동작 |
| 5 | 결과 접근과 검증 | `STUDY.md`와 Study 결과 폴더를 기존 결과 열기 흐름에 연결, 수동 시나리오 검증 | 사용 가능한 MVP |

## 11. 이후 확장 계획

| 항목 | MVP 이후 이유 |
|---|---|
| 병렬·다중 GPU Study | 리소스 스케줄링과 공정 비교 정책을 별도로 설계해야 함 |
| 모델별 batch size 자동 추천 | 비교 조건을 바꿀 수 있어 명시적 정책이 필요함 |
| 결과 자동 해석·모델 추천 | 단순 결과 생성 MVP를 검증한 후 도입 |
| Study 템플릿·이력 관리 | 반복 실험의 실제 사용 패턴을 확인한 뒤 도입 |
| 별도 Study 대시보드 | 현재 GUI의 기존 레이아웃 보존 원칙을 우선함 |

## 수용 기준

- [ ] 시작 시 단일 실행이 기본이며 기존 단일 실행이 동작한다.
- [ ] Study 모드에서 Study 이름과 2개 이상 모델을 선택할 수 있다.
- [ ] Study 모드에서 단일 arch 및 simple_cnn 전용 설정이 회색 비활성화되고 이유가 표시된다.
- [ ] 유효한 Study는 선택 모델을 순차 실행하고 로그와 모델 상태를 표시한다.
- [ ] 중단 후 같은 Study 이름으로 재실행하면 완료 run이 건너뛰어진다.
- [ ] Study 완료 후 `STUDY.md`와 개별 run 산출물에 접근할 수 있다.
