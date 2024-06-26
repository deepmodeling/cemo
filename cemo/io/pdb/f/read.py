try:
    import openmm.app as app
    import openmm.unit as unit
except ImportError: 
    import simtk.openmm.app as app 
    import simtk.unit as unit 
from ..t.PDB import PDB 
import torch


def read(f: str) -> PDB:
    obj = app.PDBFile(f)
    topo = obj.getTopology()
    coords = obj.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    return PDB(coords=coords, topo=topo)
