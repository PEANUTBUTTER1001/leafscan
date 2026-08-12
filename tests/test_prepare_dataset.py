"""데이터 인제스트의 추출본 짝 정리 검증."""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
prepare_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_dataset)


def test_reconcile_extracted_pair_deletes_unpaired_files(tmp_path):
    labeled = tmp_path / "TL"
    source = tmp_path / "TS"
    labeled.mkdir()
    source.mkdir()
    (labeled / "paired.json").write_text("{}", encoding="utf-8")
    (source / "paired.jpg").write_bytes(b"jpg")
    (labeled / "label_only.json").write_text("{}", encoding="utf-8")
    (source / "image_only.jpg").write_bytes(b"jpg")

    stems, json_only, jpg_only = prepare_dataset.reconcile_extracted_pair(labeled, source)

    assert stems == {"paired"}
    assert (json_only, jpg_only) == (1, 1)
    assert (labeled / "paired.json").exists()
    assert (source / "paired.jpg").exists()
    assert not (labeled / "label_only.json").exists()
    assert not (source / "image_only.jpg").exists()
