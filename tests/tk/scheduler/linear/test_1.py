from cemo.tk.scheduler import linear


def test():
    assert linear(0, 10) == 0.0
    assert linear(1, 10) == 0.1
    assert linear(5, 10) == 0.5
    assert linear(10, 10) == 1.0
    assert linear(11, 10) == 1.0
    assert linear(0, 0) == 1.0
    assert linear(1, 0) == 1.0
