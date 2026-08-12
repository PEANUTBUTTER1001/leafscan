"""core/config.py — config 스키마 검증·마이그레이션·환경 기록 (P9a).

기록 필드는 소급 적용이 불가능하므로 초기에 투입한다 (구현계획서 2.1).
  · schema_version / hypothesis / tags[] / parent_run / study
  · env.json  — torch·torchvision·GPU·CUDA·OS·데이터 해시·git 커밋
  · diff.json — 부모 run 과의 config 차이 (자동)
  · status.json 의 failure_kind — oom / diverged / config_error / data_error

원본 config 파일은 절대 수정하지 않는다. migrate 는 사본을 반환한다.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 2

# v2 에서 추가된 기록 필드의 기본값. v1 config 를 로드할 때 채워 넣는다.
_RECORD_DEFAULTS = {
    "schema_version": SCHEMA_VERSION,
    "hypothesis": "",
    "tags": [],
    "parent_run": None,
    "study": None,
}


def migrate(cfg: dict) -> dict:
    """v1(평면 dict) config 를 v2 로 승격한 **사본**을 반환. 원본은 건드리지 않는다.

    기존 학습 관련 키는 그대로 두므로 학습 동작(=골든 수치)은 변하지 않는다.
    """
    out = dict(cfg)  # 얕은 복사 — 원본 파일/객체 보존
    ver = int(out.get("schema_version", 1))
    if ver < 2:
        for k, default in _RECORD_DEFAULTS.items():
            out.setdefault(k, default)
        out["schema_version"] = SCHEMA_VERSION
        out["_migrated_from"] = ver
    else:
        for k, default in _RECORD_DEFAULTS.items():
            out.setdefault(k, default)
    return out


def load_config(path) -> dict:
    """config.json 을 읽어 v2 로 마이그레이션해 반환."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return migrate(raw)


# --------------------------------------------------------------------------
# 환경 스냅샷
# --------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=3,
                           cwd=Path(__file__).resolve().parent)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def file_hash(path) -> str | None:
    """파일의 sha256. 데이터 인덱스 버전 대조용."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def collect_env(cfg: dict, device: str) -> dict:
    """재현에 필요한 환경을 수집. torch 등이 없어도 죽지 않는다."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os": os.name,
        "device": device,
        "git_commit": _git_commit(),
    }
    try:
        import torch
        import torchvision
        env["torch"] = torch.__version__
        env["torchvision"] = torchvision.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
        else:
            env["gpu"] = None
    except Exception as e:  # noqa: BLE001
        env["torch_error"] = str(e)
    # 데이터 인덱스 해시 (index_csv 데이터셋일 때만 의미)
    idx = cfg.get("index_csv") or cfg.get("data_index")
    if idx:
        env["index_hash"] = file_hash(idx)
    return env


# --------------------------------------------------------------------------
# 부모 run 과의 diff
# --------------------------------------------------------------------------
_DIFF_IGNORE = {"hypothesis", "tags", "parent_run", "study", "_migrated_from",
                "device", "data_root", "num_workers"}


def config_diff(cfg: dict, parent_cfg: dict) -> dict:
    """부모 대비 바뀐 키만 {key: [old, new]}. 기록/환경성 키는 제외."""
    diff = {}
    keys = (set(cfg) | set(parent_cfg)) - _DIFF_IGNORE
    for k in sorted(keys):
        a, b = parent_cfg.get(k), cfg.get(k)
        if a != b:
            diff[k] = [a, b]
    return diff


def load_parent_config(cfg: dict, runs_dir) -> dict | None:
    parent = cfg.get("parent_run")
    if not parent:
        return None
    p = Path(runs_dir) / parent / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


# --------------------------------------------------------------------------
# 실패 분류
# --------------------------------------------------------------------------
def classify_failure(exc: BaseException) -> str:
    """예외를 failure_kind 로 분류. 콘솔·리포트가 실패 유형을 남기는 근거."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in msg or "cuda oom" in msg or "cublas" in msg:
        return "oom"
    if "nan" in msg or "inf" in msg or "diverg" in msg:
        return "diverged"
    if isinstance(exc, (KeyError, ValueError, TypeError)) and "config" in msg:
        return "config_error"
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return "data_error"
    if "index.csv" in msg or "no such file" in msg or "image" in msg:
        return "data_error"
    return "unknown"
