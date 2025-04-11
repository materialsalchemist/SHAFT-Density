from common.data_utils import *
from pymatgen.core.periodic_table import Element
from pl_modules.PygStructureDT import *


class BaseEnv:
    def __init__(self,device, ele_set='small'):
        self.all_symbol = all_chemical_symbols
        if ele_set == 'small':
            self.atoms = chemical_symbols
        elif ele_set == 'battery':
            self.atoms = battery_symbols
        elif ele_set == 'carbon':
            self.atoms = carbon_symbols
        elif ele_set == 'metal_cluster':
            self.atoms = metal_cluster
        elif ele_set == 'battery_Al':
            self.atoms = battery_symbols_Al
        else:
            self.atoms = large_chemical_symbols
        # list of Element
        self.atom_ele = [Element(atom) for atom in self.atoms]
        self.device = device
    
    def reset(self):
        raise NotImplementedError
    
    def add_atom_to(self, structure, atomidx, coord_pos):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

    def structures2repr(self, structure=None, for_proxy=False):
        raise NotImplementedError



    
