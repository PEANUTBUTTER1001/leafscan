# 05. wandb 연동 (실험 추적)

> LeafScan Lab 의 학습 기록을 [Weights & Biases](https://wandb.ai)(wandb) 서버에
> 실시간으로 남기는 기능의 구현 설명과 팀원 셋업 가이드.
> 프로젝트: **leafscan-lab** · 적용일: 2026-08-14

---

## 1. 설계 원칙

**추적이 학습을 절대 방해하지 않는다.**

| 상황 | 동작 |
|---|---|
| wandb 미설치 | 조용히 건너뜀 — 학습 정상 완주 |
| 설치했지만 미로그인 | **offline 모드**로 로컬 `wandb/` 폴더에 기록 (이후 `wandb sync` 로 업로드 가능) |
| 로그인 완료 | **online 모드**로 서버에 실시간 기록 |
| 기록 중 오류/네트워크 단절 | 해당 호출만 무시 — 학습 계속 |

미로그인 상태에서 `wandb.init()` 이 API 키 입력 프롬프트를 띄우면 GUI 가 띄운
워커 프로세스가 영원히 멈추기 때문에, **init 전에 로그인 여부를 직접 확인**해서
모드를 정하는 것이 구현의 핵심이다.

## 2. 전체 흐름

```
GUI (lab_gui.py)                train_worker.py                wandb
┌───────────────────┐           ┌────────────────────┐
│ 고급 › wandb 체크   │─ config →│ Tracker 초기화       │── online ──→ 서버 실시간 기록
│   (use_wandb)     │           │  (core/tracking.py) │── offline ─→ 로컬 wandb/ 폴더
│                   │← 이벤트 ──│ · epoch마다 지표      │
│ "wandb 보기" 버튼  │  wandb.json│ · 종료 시 최종 요약   │
└───────────────────┘           └────────────────────┘
```

## 3. 파일별 구현 내용

### core/tracking.py — 추적 래퍼 (핵심)

- `Tracker` 클래스가 MLflow/wandb 를 감싼다. 모든 호출이 try/except 로 보호되어
  실패 시 조용히 no-op 이 된다.
- `_wandb_logged_in()` — `WANDB_API_KEY` 환경변수 또는 `~/.netrc`·`~/_netrc`(Windows)
  파일에서 API 키 존재를 확인한다. 없으면 offline 모드로 전환해 **워커가 로그인
  프롬프트에서 멈추는 것을 막는다.**
- `wandb.init(project="leafscan-lab", name=<run 디렉터리명>, group=<study 이름>, config=<전체 설정>)`
  — run 이름이 로컬 `runs/` 폴더명과 일치하고, Study 멤버 run 들은 group 으로 묶여
  wandb 화면에서 한 묶음으로 비교된다.
- `WANDB_SILENT=true` — 워커 stdout 은 GUI 가 JSON 라인으로 파싱하므로
  wandb 의 안내 출력이 섞이지 않게 차단한다.
- 제공 API: `log_metrics(dict, step)` · `log_summary(dict)` · `log_artifact(path)` ·
  `run_url`(온라인 run 의 웹 URL) · `messages`(GUI 로그용 상태 문구)

### train_worker.py — 기록 시점

| 시점 | 기록 내용 |
|---|---|
| 학습 시작 | Tracker 생성 → 상태 문구 로그 → run URL 을 `{"event":"wandb","url":...}` 이벤트로 GUI에 알리고 run 폴더에 `wandb.json` 으로 영구 저장 |
| epoch 마다 | `loss` `val_loss` `acc` `val_acc` + 멀티헤드면 `val_acc_crop` `val_acc_stage` |
| 학습 종료 | run summary 에 `stage_macro_f1` `stage_primary_recall` `crop_accuracy` `best_val_loss` `best_epoch` `params` `elapsed_sec` 등 + `metrics.json` 파일 첨부 |

### lab_gui.py — 사용자 인터페이스

- **고급 › "wandb 실험 추적" 체크박스** (기본 켜짐) → config 의 `use_wandb` 로 전달.
  단일 실행·Study 모드 공통.
- **상단 "wandb 보기" 버튼** — 상황에 따라 여는 페이지가 다르다:
  1. 결과 화면에서 run 을 보고 있으면 → 그 run 의 wandb 페이지 (`run 폴더/wandb.json`)
  2. 학습 진행 중이면 → 지금 돌고 있는 run 의 실시간 차트
  3. 그 외 → 가장 최근 기록에서 만든 `leafscan-lab` 프로젝트 대시보드
  4. 기록이 하나도 없으면 안내 메시지

### 설정 파일

- `configs/base_leafscan.json` — `"use_wandb": true`, `"wandb_project": "leafscan-lab"`
  추가. CLI(`run_study.py`) 실행도 자동 기록된다.
- `requirements.txt` — `wandb` 추가.

## 4. 팀원 셋업 가이드 (PC 당 최초 1회)

1. **패키지 설치**

   ```
   pip install wandb
   ```

2. **로그인** — https://wandb.ai/authorize 에서 API 키를 복사한 뒤:

   ```
   wandb login
   ```

   붙여넣어도 화면에 **아무 글자도 안 보이는 게 정상**이다 (비밀번호식 숨김 입력).
   cmd 에서는 우클릭이 붙여넣기 → 그대로 Enter.
   그래도 안 되면 키를 명령에 직접 붙인다: `wandb login --relogin <API키>`

3. **사용** — LeafScan Lab 에서 평소처럼 학습을 실행하면 끝.
   로그 창에 `wandb 추적 활성 (online) → URL` 이 뜨면 연동 성공이며,
   "wandb 보기" 버튼 또는 https://wandb.ai 의 `leafscan-lab` 프로젝트에서 확인한다.

### 로그인 없이 학습한 경우 (offline 기록 업로드)

```
wandb login
wandb sync wandb/offline-run-<날짜_ID>
```

## 5. 문제 해결

| 증상 | 원인·해결 |
|---|---|
| 로그에 wandb 문구가 아예 없음 | wandb 미설치 (`pip install wandb`) 또는 고급 › 체크 해제 상태 |
| `offline — 로그인 안 됨` 으로 나옴 | `wandb login` 후 다음 학습부터 online. 이미 쌓인 offline run 은 `wandb sync` 로 업로드 |
| run 이 개인 계정으로 들어감 | 현재 로그인 계정의 기본 entity 로 기록된다. 팀 계정으로 모으려면 wandb 팀(entity)에 가입 — config 에 `wandb_entity` 키를 받는 확장은 tracking.py 의 `wandb.init` 에 `entity=` 한 줄이면 된다 |
| 학습이 wandb 때문에 느려지는 의심 | 기록은 비동기이며 epoch 당 호출 1회 수준이라 영향이 거의 없다. 확실히 끄려면 고급 › 체크 해제 |
