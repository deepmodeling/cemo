from cemo.io import pdb
import filecmp

def test():
    f_in = "input/7BCQ.pdb"
    f_out = "tmp/test1_out.pdb"
    f_expect = "expect/test1_out.pdb"
    obj = pdb.read(f_in)
    print(type(obj))
    print(type(obj.coords))
    print(obj.coords.shape)
    print(type(obj.topo))
    pdb.write(f_out, obj, keepIds=True)
    assert filecmp.cmp(f_out, f_expect, shallow=False)
