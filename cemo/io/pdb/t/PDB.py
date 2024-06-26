from dataclasses import dataclass
import numpy

try:
    from openmm.app.topology import Topology
except ImportError: 
    from simtk.openmm.app.topology import Topology

@dataclass
class PDB:
    coords: numpy.ndarray
    topo: Topology
