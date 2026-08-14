"""train_worker 의 진행 이벤트 스로틀 검증 — 학습을 돌리지 않는다.

긴 무출력 구간(라벨 수집·epoch 내부·예측 수집)에서 진행 상황을 알리되,
초당 수십 줄이 쏟아지지 않도록 스로틀이 걸려 있어야 한다.

    python -m pytest tests/test_train_worker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_worker import ProgressEmitter


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += sec


def make(interval=1.0):
    sent = []
    clock = FakeClock()
    prog = ProgressEmitter(interval=interval, clock=clock,
                           sink=lambda **kw: sent.append(kw))
    return prog, sent, clock


def test_first_step_always_emits():
    prog, sent, _clock = make()
    assert prog.step("작업", 1, 100) is True
    assert sent[0]["event"] == "progress"
    assert sent[0]["phase"] == "작업"
    assert (sent[0]["done"], sent[0]["total"]) == (1, 100)


def test_throttled_between_first_and_last():
    prog, sent, clock = make(interval=1.0)
    prog.step("작업", 1, 100)          # 처음 — 발행
    assert prog.step("작업", 2, 100) is False   # 0초 경과 — 억제
    clock.advance(0.5)
    assert prog.step("작업", 3, 100) is False   # 아직 1초 미만
    clock.advance(0.6)
    assert prog.step("작업", 4, 100) is True    # 1.1초 경과 — 발행
    assert [s["done"] for s in sent] == [1, 4]


def test_last_step_always_emits():
    """완료 시점은 스로틀과 무관하게 항상 보인다."""
    prog, sent, _clock = make(interval=1.0)
    prog.step("작업", 1, 10)
    assert prog.step("작업", 10, 10) is True    # 0초 경과지만 마지막
    assert [s["done"] for s in sent] == [1, 10]


def test_phase_resets_throttle():
    """단계가 바뀌면 곧바로 한 번 보여준다."""
    prog, sent, _clock = make(interval=1.0)
    prog.step("A", 1, 100)
    assert prog.step("A", 2, 100) is False
    prog.phase("B")
    assert sent[-1] == {"event": "phase", "name": "B"}
    assert prog.step("B", 1, 100) is True       # 새 단계는 즉시 발행


def test_extra_fields_pass_through():
    """진행바 세분화용 epoch_frac 이 그대로 실려야 한다."""
    prog, sent, _clock = make()
    prog.step("epoch 3 학습", 5, 10, epoch_frac=2.25)
    assert sent[0]["epoch_frac"] == 2.25


def test_unknown_total_still_throttles():
    prog, sent, clock = make(interval=1.0)
    prog.step("총량모름", 1)
    assert prog.step("총량모름", 2) is False
    clock.advance(1.5)
    assert prog.step("총량모름", 3) is True
    assert [s["total"] for s in sent] == [0, 0]


if __name__ == "__main__":
    for fn in (test_first_step_always_emits, test_throttled_between_first_and_last,
               test_last_step_always_emits, test_phase_resets_throttle,
               test_extra_fields_pass_through, test_unknown_total_still_throttles):
        fn(); print(f"  ✓ {fn.__name__}")
    print("[train_worker 진행 이벤트] 전체 통과")
