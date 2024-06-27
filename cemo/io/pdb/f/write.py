try:
    import openmm.app as app
except ImportError: 
    import simtk.openmm.app as app 
from ..t.PDB import PDB


def write(fname: str, obj: PDB, keepIds=True):
    with open(fname, "w") as OUT:
        app.PDBFile.writeFile(
            obj.topo,
            obj.coords.numpy(),
            file=OUT,
            keepIds=keepIds,
            extraParticleIdentifier="",
        )
