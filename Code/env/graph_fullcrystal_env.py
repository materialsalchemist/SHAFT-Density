from env.crystal_env import BaseEnv
from pl_modules.structure import CrystalStructureCData
from pl_modules.megnet.megnetdataset import MEGNetDataset, collate_fn
from dgl.dataloading import GraphDataLoader
import math
from pl_modules.graphs import PygGraph as CPygGraph
from common.alignn import get_figshare_model
from matformer.models.pyg_att import Matformer
import torch
import numpy as np
from pl_modules.PygStructureDT import PygStructureDT ,atoms_to_graph
from torch.utils.data import DataLoader
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from matformer.models.pyg_att import Matformer
from alignn.data import get_torch_dataset
from torch_geometric.data import Data
from matgl.ext.pymatgen import Structure2Graph

class HierGraphCrystalEnv(BaseEnv):
    def __init__(self,device,pretrain_model, ele_set='small',batch_size=32,max_atom=200):
        super().__init__(device,ele_set)
        if pretrain_model == 'matformer':
            self.pretrain_model = Matformer()
            self.pretrain_model.load_state_dict(torch.load('./pretrain_matformer/matformer_mp_best_model_497_neg_mae=-0.0221.pt'))
            self.pretrain_model.to(self.device)
        elif pretrain_model == 'alignn':
            self.pretrain_model = get_figshare_model()
        else:
            raise NotImplementedError
        
        self.pretrain_model_name = pretrain_model
        self.batch_size = batch_size
        self.max_atom = max_atom

    def set_req(self,req_config):
        self.req_config = req_config
    
    def reset(self):
        # self.structure = CrystalStructureCData()
        self.structures = []
        for i in range(self.batch_size):
            createa = CrystalStructureCData(self.req_config)
            self.structures.append(createa)
        self.structures = np.asarray(self.structures,dtype=object)


    def set_spacegroup_of(self, structure:CrystalStructureCData, spacegroup):
        structure.set_spacegroup(spacegroup=spacegroup)
        return structure

    def set_lattice_of(self, structure:CrystalStructureCData, lattice):
        l1s = lattice[0]
        l2s = lattice[1]
        l3s = lattice[2]
        a1s = math.degrees(lattice[3])
        a2s = math.degrees(lattice[4])
        a3s = math.degrees(lattice[5])        
        lattice = np.array([l1s,l2s,l3s,a1s,a2s,a3s])
        structure.set_lattice(lattice=lattice)
        return structure

    def add_atom_to(self, structure:CrystalStructureCData, atom):
        new_structure = structure.copy()
        fracs = atom[:3]
        atype = int(atom[3])
        assert (atype >= 0) and (atype <= len(self.atoms)), "unknown elements"
        atomidx = self.all_symbol.index(self.atoms[atype])
        new_structure.add_atom(atomidx=atomidx,atom=self.atom_ele[atype],coord_pos=fracs)
        return new_structure

    def apply_actions_to(self, structure:CrystalStructureCData, action):
        act = action.cpu().detach().numpy()
        new_structure = structure.copy()
        sg_act = act[0]
        lattice_act = act[1:7]
        atom_act = act[7:12]
        new_structure = self.set_spacegroup_of(new_structure,sg_act)
        new_structure = self.set_lattice_of(new_structure,lattice_act)
        new_structure = self.add_atom_to(new_structure,atom_act)
        return new_structure

    def step(self, actions, step, min_stop):
        new_structs = []
        for idx,struct in enumerate(self.structures):
            if (struct.complete == False) and torch.all(actions[idx] != -float("inf")):
                new_struct = self.apply_actions_to(struct,actions[idx][1:])
                if (len(new_struct.structure) > self.max_atom): # if after step, the number of atom 
                    # print('Max atom ',len(new_struct.structure),' exceeds threshold', self.max_atom)
                    new_struct.complete = True
                    new_struct = struct
            else: 
                struct.complete = True
                new_struct =  struct
            new_structs.append(new_struct)
        self.structures = new_structs
        self.structures = np.asarray(self.structures,dtype=object)


    def get_matformer_pretrain_feature(self, structures,n_trajectories=100):
        if len(structures[0].atomic_numbers) != 0:
            graphs = [atoms_to_graph(a.structure,include_coor=False) for a in structures]
            data= PygStructureDT(
                        graphs,
                        atom_features='cgcnn',
                        line_graph=True,
                        neighbor_strategy='k-nearest',
                    )
            collate_fn = data.collate
            test_loader = DataLoader(
                    data,
                    batch_size=n_trajectories,
                    shuffle=False,
                    collate_fn=collate_fn,
                    drop_last=False,
                    num_workers=8,
                    pin_memory=False,
                )
            g, _ = next(iter(test_loader))
            _, pretrain_feature = self.pretrain_model(g.to(self.device),True)
        else:
            pretrain_feature = torch.zeros(size=(len(structures),128),device=self.device,dtype=torch.float32)
        return pretrain_feature

    def get_alignn_pretrain_feature(self, structures,n_trajectories=100):
        if len(structures[0].atomic_numbers) != 0:
            atoms_array = [JarvisAtomsAdaptor.get_atoms(mo.structure) for mo in structures]
            mem = []
            for i, ii in enumerate(atoms_array):
                info = {}
                info["atoms"] = ii.to_dict()
                info["prop"] = -9999  # place-holder only
                info["jid"] = str(i)
                mem.append(info)

            test_data = get_torch_dataset(
                dataset=mem,
                target="prop",
                neighbor_strategy='k-nearest',
                atom_features="cgcnn",
                use_canonize=True,
                line_graph=True,
            )

            collate_fn = test_data.collate_line_graph
            test_loader = DataLoader(
                test_data,
                batch_size=n_trajectories,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
                num_workers=0,
                pin_memory=False,
            )
            g, lg, target = next(iter(test_loader))

            pretrain_feature = (
                        self.pretrain_model([g.to(self.device), lg.to(self.device)])
                    )
        else:
            pretrain_feature = torch.zeros(size=(len(structures),256),device=self.device,dtype=torch.float32)
        return pretrain_feature

    def structures2repr(self, structures=None, use_pretrain=False,n_trajectories=100, mask=None):
        if structures is None:
            structures = self.structures
        if len(self.structures[0].frac_coords)==0:
            return (None,None,None,None)
        if mask is not None:
            structures = self.structures[mask]
        if use_pretrain:
            if self.pretrain_model_name == 'alignn':
                pretrain_reps = self.get_alignn_pretrain_feature(structures,n_trajectories=n_trajectories)
            elif self.pretrain_model_name =='matformer':
                pretrain_reps = self.get_matformer_pretrain_feature(structures,n_trajectories=n_trajectories)
        else:
            pretrain_reps = np.zeros((len(structures),128))

        lattice_reps = self.structures2latticereps(structures)

        structure_graphreps = self.structs2batch(structures=structures,
                                                      linegraph=False,
                                                      atom_features='atomic_number',
                                                      include_coor=True,
                                                      n_trajectories=n_trajectories)
        spacegroup_reps = self.structures2spacegroupreps(structures)
    
        return (structure_graphreps, 
                torch.tensor(spacegroup_reps, device=self.device,dtype=torch.int),
                torch.tensor(lattice_reps, device=self.device,dtype=torch.float32),
                pretrain_reps
                )
    
    def structures2spacegroupreps(self, structures:CrystalStructureCData):
        features = []
        for structure in structures:
            features.append(structure.spacegroup-1)
        return np.array(features)


    def structures2latticereps(self, structures):
        # Implement state encoding
        features = []
        for structure in structures:
            lengths = structure.lattice[:3]
            angles = structure.lattice[3:] 
            feature = np.array([np.sin(angles[0]* np.pi/180),np.cos(angles[0]* np.pi/180),
                                 np.sin(angles[1]* np.pi/180),np.cos(angles[1]* np.pi/180),
                                 np.sin(angles[2]* np.pi/180),np.cos(angles[2]* np.pi/180),
                                 lengths[0],lengths[1],lengths[2],
                                 structure.spacegroup])
            norm_feature = (feature - feature.mean())/feature.std()
            features.append(norm_feature)
        return np.array(features) 
    

    def atoms_to_graph(self,atoms:Structure,linegraph=False,atom_features='atomic_number',include_coor=True):
        raise NotImplementedError

    def structs2batch(self, structures,linegraph=False,atom_features='atomic_number',include_coor=True,n_trajectories=100):
        raise NotImplementedError
    

class HierGraphMEGNetCrystalEnv(HierGraphCrystalEnv):
    def __init__(self,device,pretrain_model, ele_set='small',batch_size=32,max_atom=200):
        super().__init__(device,pretrain_model, ele_set,batch_size,max_atom=max_atom)
        self.labels = 0
        elem_list = tuple(self.atoms)
        self.converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
        self.initial = 0.0
        self.final = 5.0
        self.num_centers = 100
        self.width = 0.5

    def structs2batch(self, structures,linegraph=False,atom_features='atomic_number',include_coor=True,n_trajectories=100):
        num_graphs = len(structures)
        structures = [struct.structure for struct in structures]
        eform_per_atom = [0.0]*num_graphs

        dataset = MEGNetDataset(
            structures=structures,
            converter=self.converter,
            labels={"Eform": eform_per_atom},
            initial=0.0,
            final=5.0,
            num_centers=100,
            width=0.5,
            )
        
        loader = GraphDataLoader(dataset, shuffle=False, collate_fn=collate_fn, 
                                 batch_size=num_graphs, num_workers=1,)
        
        g, lattice, state_attr, label = (next(iter(loader)))
        node_feat = g.ndata["node_type"]
        edge_feat = g.edata["edge_attr"]
        return (g.to(self.device), edge_feat.to(self.device), node_feat.to(self.device), state_attr.to(self.device))
    


class HierDensityGridCrystalEnv(HierGraphMEGNetCrystalEnv):
    def __init__(self,device,pretrain_model, ele_set='small',batch_size=32,max_atom=200):
        super().__init__(device,pretrain_model, ele_set,batch_size,max_atom=max_atom)
        self.labels = 0
        elem_list = tuple(self.atoms)
        self.converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
        self.initial = 0.0
        self.final = 5.0
        self.num_centers = 100
        self.width = 0.5
        self.min_density_point = None
    
    def calculate_density_grid(self, structure:Structure, cutoff=4.0, divisions = [30,30,30]):
        supercell = structure.copy()
        supercell = structure.make_supercell([3,3,3])
        supercell_matrix = torch.tensor(supercell.lattice.matrix, dtype=torch.float32).cuda()
        divisions = torch.tensor(divisions).cuda()

        # Generate grid points for i, j, k
        i_vals = torch.arange(0, divisions[0], dtype=torch.float32, device='cuda')
        j_vals = torch.arange(0, divisions[1], dtype=torch.float32, device='cuda')
        k_vals = torch.arange(0, divisions[2], dtype=torch.float32, device='cuda')

        # Create a meshgrid of i, j, k values
        i_grid, j_grid, k_grid = torch.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        n = torch.zeros((int(divisions[0]), int(divisions[1]), int(divisions[2])), dtype=torch.float32).cuda()

        # Normalize the grid to get f = [i / divisions[0], j / divisions[1], k / divisions[2]]
        f_grid = torch.stack([
            i_grid / divisions[0],
            j_grid / divisions[1],
            k_grid / divisions[2]
        ], dim=-1)  # Shape will be (divisions[0], divisions[1], divisions[2], 3)

        # Reshape for matrix multiplication
        f_grid_flat = f_grid.view(-1, 3)  # Flatten to (N, 3), where N = total number of points

        # Compute r = f * supercell.lattice.matrix (batch matrix multiplication)
        r_flat = torch.matmul(f_grid_flat, supercell_matrix.T)  # Shape will be (N, 3)

        # Reshape r_flat back to 3D grid
        r_grid = r_flat.view(divisions[0], divisions[1], divisions[2], 3)

        cart_coords = torch.tensor(supercell.cart_coords, dtype=torch.float32).cuda()

        # Step 9: Loop through `R` in `supercell.cart_coords` and compute distances and density in parallel
        cutoff = torch.tensor(4.0, dtype=torch.float32).cuda()

        for R in cart_coords:
            # Compute distances between each `r` and the current `R`
            distances = torch.norm(r_grid - R, dim=-1)  # Shape (divisions[0], divisions[1], divisions[2])
            
            # Apply cutoff condition and compute density where distances are within the cutoff
            mask = distances <= cutoff
            density_values = torch.exp(-distances[mask]**2)  # Applying density function

            # Add the density values to the grid `n`
            n[mask] += density_values
        n = n[10:20, 10:20, 10:20]
        return n

    def caculate_low_density_point_batch(self, structures):
        divisions = [30,30,30]
        self.divisions = [divs/3 for divs in divisions]
        cutoff=4.0
        # structures = [s.structure for s in structures]
        density_maps = [self.calculate_density_grid(s.structure,divisions=divisions,cutoff=cutoff) for s in structures]
        ps = [torch.where(density_map == torch.min(density_map)) for density_map in density_maps]
        idx = [torch.randint(low=0,high=len(p[0]),size=(1,)) for p in ps]
        # self.min_density_point = [torch.tensor([p[0][0]/(divisions[0]/3),p[1][0]/(divisions[0]/3),p[2][0]/(divisions[0]/3)]).cuda() for p in ps]
        ps = torch.stack([torch.cat([p[0][id]/(self.divisions[0]),p[1][id]/(self.divisions[1]),p[2][id]/(self.divisions[2])]) for id,p in zip(idx,ps)])
        return ps

    def add_atom_to(self, structure:CrystalStructureCData, atom, min_density_point):
        new_structure = structure.copy()
        if min_density_point is None:
            fracs = atom[:3]
        else:
            fracs = atom[:3]*([1/dv for dv in self.divisions])
            fracs = min_density_point + fracs
        atype = int(atom[3])
        assert (atype >= 0) and (atype <= len(self.atoms)), "unknown elements"
        atomidx = self.all_symbol.index(self.atoms[atype])
        new_structure.add_atom(atomidx=atomidx,atom=self.atom_ele[atype],coord_pos=fracs)
        return new_structure
    
    def apply_actions_to(self, structure:CrystalStructureCData, action, min_density_point, step):
        act = action.cpu().detach().numpy()
        new_structure = structure.copy()
        sg_act = act[0]
        lattice_act = act[1:7]
        atom_act = act[7:12]
        if step == 0:
            new_structure = self.set_spacegroup_of(new_structure,sg_act)
            new_structure = self.set_lattice_of(new_structure,lattice_act)
        new_structure = self.add_atom_to(new_structure,atom_act, min_density_point)
        return new_structure

    def step(self, actions, step, min_stop):
        new_structs = []
        not_complete_struck_mask = [not s.complete for s in self.structures]
        # print('not_complete_struck_mask',not_complete_struck_mask)
        structures = self.structures[not_complete_struck_mask]
        for idx,struct in enumerate(structures):
            if torch.all(actions[idx] != -float("inf")):
                if self.min_density_point is None:
                    new_struct = self.apply_actions_to(struct,actions[idx][1:], None, step)
                else:
                    new_struct = self.apply_actions_to(struct,actions[idx][1:], self.min_density_point[idx], step)
                if (len(new_struct.structure) > self.max_atom): 
                    new_struct.complete = True
            else: 
                struct.complete = True
                new_struct = struct
            new_structs.append(new_struct)

        self.structures[not_complete_struck_mask] = new_structs
        self.structures = np.asarray(self.structures,dtype=object)

    def reset(self):
        self.structures = []
        for i in range(self.batch_size):
            createa = CrystalStructureCData(self.req_config)
            self.structures.append(createa)
        self.structures = np.asarray(self.structures,dtype=object)
        self.min_density_point = None

    def structures2repr(self, structures=None, use_pretrain=False,n_trajectories=100, mask=None):
        if structures is None:
            structures = self.structures
        if len(self.structures[0].frac_coords)==0:
            return (None,None,None,None)
        if mask is not None:
            structures = self.structures[mask]
        if use_pretrain:
            if self.pretrain_model_name == 'alignn':
                pretrain_reps = self.get_alignn_pretrain_feature(structures,n_trajectories=n_trajectories)
            elif self.pretrain_model_name =='matformer':
                pretrain_reps = self.get_matformer_pretrain_feature(structures,n_trajectories=n_trajectories)
        else:
            pretrain_reps = np.zeros((len(structures),128))

        lattice_reps = self.structures2latticereps(structures)
        structure_graphreps = self.structs2batch(structures=structures,
                                                      linegraph=False,
                                                      atom_features='atomic_number',
                                                      include_coor=True,
                                                      n_trajectories=n_trajectories)
        spacegroup_reps = self.structures2spacegroupreps(structures)

        min_density_reps = self.caculate_low_density_point_batch(structures)

        self.min_density_point = min_density_reps.cpu().detach().numpy()
    
        return (structure_graphreps, 
                torch.tensor(spacegroup_reps, device=self.device,dtype=torch.int),
                torch.tensor(lattice_reps, device=self.device,dtype=torch.float32),
                pretrain_reps,
                torch.tensor(min_density_reps, device=self.device,dtype=torch.float32)
                )