"""core/tracking.py — 실험 추적 래퍼 (P8).

MLflow 우선, wandb 선택. **둘 다 try/except** 로 감싸 미설치 상태에서도
학습이 정상 완주한다 (완료 판정). run 디렉터리명 = MLflow run = wandb run 으로 통일.
"""


class Tracker:
    """미설치·오류 시 조용히 no-op 이 되는 추적 래퍼."""

    def __init__(self, run_name, cfg, enable=True):
        self.run_name = run_name
        self.mlflow = None
        self.wandb = None
        self._active = False
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
        except Exception:  # noqa: BLE001 — 미설치/서버없음 등 모두 무시
            self.mlflow = None

    def _try_wandb(self, run_name, cfg):
        if not cfg.get("use_wandb"):
            return
        try:
            import wandb
            wandb.init(name=run_name, project=cfg.get("study") or "leafscan",
                       config=cfg, reinit=True)
            self.wandb = wandb
            self._active = True
        except Exception:  # noqa: BLE001
            self.wandb = None

    def log_metrics(self, metrics: dict, step=None):
        if self.mlflow:
            try:
                self.mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float))}, step=step)
            except Exception:  # noqa: BLE001
                pass
        if self.wandb:
            try:
                self.wandb.log(metrics, step=step)
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
