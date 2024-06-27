from cemo.tk.scheduler import jump


def test():
    assert jump(5, 5, True) == 1.0
    assert jump(4, 5, True) == 0.0
    assert jump(5, 5, False) == 0.0
    assert jump(6, 5, False) == 1.0
    assert jump(-1, 0, True) == 0.0
    assert jump(-1, 0, False) == 0.0
    assert jump(0, 0, True) == 1.0
    assert jump(0, 0, False) == 0.0
