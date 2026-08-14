"""core/tracking.py — 실험 추적 래퍼 (P8).

MLflow 우선, wandb 선택. **둘 다 try/except** 로 감싸 미설치 상태에서도
학습이 정상 완주한다 (완료 판정). run 디렉터리명 = MLflow run = wandb run 으로 통일.

wandb 는 config 의 use_wandb 로 켠다 (GUI: 고급 › wandb 체크박스).
  · 로그인 상태(wandb login / WANDB_API_KEY) → online 으로 서버에 실시간 기록
  · 미로그인 → offline 으로 로컬(wandb/ 폴더)에 기록하고, 나중에
    `wandb sync` 로 서버에 올릴 수 있다. **미로그인이어도 학습이 멈추지 않는다.**
  · study 이름이 있으면 wandb group 으로 묶여 run 비교가 쉽다.
"""

import os
from pathlib import Path


def _wandb_logged_in():
    """wandb API 키 존재 여부. 로그인 프롬프트로 워커가 멈추는 것을 막는다."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    import netrc
    for fname in (".netrc", "_netrc"):     # Windows 는 _netrc 를 쓰기도 한다
        try:
            auth = netrc.netrc(str(Path.home() / fname)).authenticators("api.wandb.ai")
            if auth and auth[2]:
                return True
        except Exception:  # noqa: BLE001 — 파일 없음/파싱 실패 모두 무시
            continue
    return False


class Tracker:
    """미설치·오류 시 조용히 no-op 이 되는 추적 래퍼."""

    def __init__(self, run_name, cfg, enable=True):
        self.run_name = run_name
        self.mlflow = None
        self.wandb = None
        self._active = False
        self.messages = []   # 워커가 GUI 로그에 그대로 내보낼 상태 문구
        self.run_url = None  # wandb online run 의 웹 URL (GUI 바로가기 버튼용)
        if not enable:
            return
        self._try_mlflow(run_name, cfg)
        self._try_wandb(run_name, cfg)

    def _try_mlflow(self, run_name, cfg):
        try:
            import mlflow
            mlflow.set_experiment(cfg.get("study") or "leafscan")
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: v for k, v in cfg.items()
                               if isinstance(v, (int, float, str, bool))})
            self.mlflow = mlflow
            self._active = True
            self.messages.append("MLflow 추적 활성")
        except Exception:  # noqa: BLE001 — 미설치/서버없음 등 모두 무시
            self.mlflow = None

    def _try_wandb(self, run_name, cfg):
        if not cfg.get("use_wandb"):
            return
        try:
            # 워커 stdout 은 GUI 가 JSON 라인으로 파싱하므로 wandb 출력을 줄인다
            os.environ.setdefault("WANDB_SILENT", "true")
            import wandb

            mode = os.environ.get("WANDB_MODE")   # 사용자가 정했으면 존중
            if not mode:
                mode = "online" if _wandb_logged_in() else "offline"

            run = wandb.init(
                project=cfg.get("wandb_project") or "leafscan-lab",
                name=run_name,
                group=cfg.get("study") or None,     # study 단위로 run 을 묶는다
                job_type="train",
                tags=[t for t in (cfg.get("tags") or []) if isinstance(t, str)],
                config=cfg,
                mode=mode,
            )
            self.wandb = wandb
            self._active = True
            if mode == "online":
                url = getattr(run, "url", None)
                self.run_url = url
                self.messages.append(f"wandb 추적 활성 (online){' → ' + url if url else ''}")
            else:
                self.messages.append(
                    "wandb 추적 활성 (offline — 로그인 안 됨). 로컬 wandb/ 폴더에 "
                    "기록되며 `wandb login` 후 `wandb sync wandb/latest-run` 으로 "
                    "서버에 올릴 수 있습니다.")
        except Exception:  # noqa: BLE001
            self.wandb = None

    def log_metrics(self, metrics: dict, step=None):
        """epoch 단위 지표. 숫자만 골라 기록한다."""
        nums = {k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float))}
        if self.mlflow:
            try:
                self.mlflow.log_metrics(nums, step=step)
            except Exception:  # noqa: BLE001
                pass
        if self.wandb:
            try:
                self.wandb.log(nums, step=step)
            except Exception:  # noqa: BLE001
                pass

    def log_summary(self, summary: dict):
        """학습 종료 후 최종 지표(macro-F1 등)를 run 요약으로 남긴다."""
        nums = {k: float(v) for k, v in summary.items()
                if isinstance(v, (int, float))}
        if self.mlflow:
            try:
                self.mlflow.log_metrics(nums)
            except Exception:  # noqa: BLE001
                pass
        if self.wandb:
            try:
                self.wandb.run.summary.update(nums)
            except Exception:  # noqa: BLE001
                pass

    def log_artifact(self, path):
        """metrics.json 같은 결과 파일을 run 에 첨부한다 (가능한 백엔드만)."""
        p = str(path)
        if self.mlflow:
            try:
                self.mlflow.log_artifact(p)
            except Exception:  # noqa: BLE001
                pass
        if self.wandb:
            try:
                self.wandb.save(p, policy="now")
            except Exception:  # noqa: BLE001
                pass

    def end(self):
        if self.mlflow:
            try:
                self.mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass
        if self.wandb:
            try:
                self.wandb.finish()
            except Exception:  # noqa: BLE001
                pass

    @property
    def active(self):
        return self._active
