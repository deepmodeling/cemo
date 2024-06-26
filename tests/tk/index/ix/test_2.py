import torch
from cemo.tk.index import ix


def test_indexing_1():
    x = torch.arange(9).view(3, 3)
    id1 = torch.tensor([0, 2])
    id2 = torch.tensor([0, 1])
    idx = ix([id1, id2])
    result = x[idx]
    expect = x[[0, 2], 0:2]
    assert torch.allclose(result, expect, atol=1e-6, rtol=0.)


def test_indexing_2():
    x = torch.arange(90).view(10, 3, 3)
    id1 = torch.tensor([1, 2, 3, 4])
    id2 = torch.tensor([0, 2])
    id3 = torch.tensor([0, 1, 2])
    idx = ix([id1, id2, id3])
    result = x[idx]
    expect = x[1:5, [0, 2], :3]
    assert torch.allclose(result, expect, atol=1e-6, rtol=0.)
