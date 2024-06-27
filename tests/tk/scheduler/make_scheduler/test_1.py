import pytest
from cemo.tk.scheduler import make_scheduler, jump, linear


def test_make_scheduler_jump():
    scheduler = make_scheduler('jump', 10)
    assert scheduler(5) == jump(5, 10)


def test_make_scheduler_jump_inclusive():
    scheduler = make_scheduler('jump', 10, False)
    assert scheduler(10) == jump(10, 10, False)


def test_make_scheduler_linear():
    scheduler = make_scheduler('linear', 10)
    assert scheduler(5) == linear(5, 10)


def test_make_scheduler_unknown():
    with pytest.raises(ValueError) as e:
        make_scheduler('unknown', 10)
    assert str(e.value) == 'Unknown scheduler: unknown'
