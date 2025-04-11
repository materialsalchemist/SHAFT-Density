"""Module to generate networkx graphs."""
"""Implementation based on the template of ALIGNN."""
from multiprocessing.context import ForkContext
from re import X
import numpy as np
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes

# from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
import itertools

try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass

# pyg dataset

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))
            
    return edges



def pair_nearest_neighbor_edges(
        atoms=None,
        pair_wise_distances=6,
        use_lattice=False,
        use_angle=False,
):
    """Construct pairwise k-fully connected edge list."""
    smallest = pair_wise_distances
    lattice_list = torch.as_tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]).float()

    lattice = torch.as_tensor(atoms.lattice_mat).float()
    pos = torch.as_tensor(atoms.cart_coords)
    atom_num = pos.size(0)
    lat = atoms.lattice
    radius_needed = min(lat.a, lat.b, lat.c) * (smallest / 2 - 1e-9)
    r_a = (np.floor(radius_needed / lat.a) + 1).astype(np.int)
    r_b = (np.floor(radius_needed / lat.b) + 1).astype(np.int)
    r_c = (np.floor(radius_needed / lat.c) + 1).astype(np.int)
    period_list = np.array([l for l in itertools.product(*[list(range(-r_a, r_a + 1)), list(range(-r_b, r_b + 1)), list(range(-r_c, r_c + 1))])])
    period_list = torch.as_tensor(period_list).float()
    n_cells = period_list.size(0)
    offset = torch.matmul(period_list, lattice).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offset).transpose(0, 1).contiguous()
    dist = (pos.unsqueeze(1).unsqueeze(1) - expand_pos.unsqueeze(0))  # [n, 1, 1, 3] - [1, n, n_cell, 3] -> [n, n, n_cell, 3]
    dist2, index = torch.sort(dist.norm(dim=-1), dim=-1, stable=True)
    max_value = dist2[:, :, smallest - 1]  # [n, n]
    mask = (dist.norm(dim=-1) <= max_value.unsqueeze(-1))  # [n, n, n_cell]
    shift = torch.matmul(lattice_list, lattice).repeat(atom_num, 1)
    shift_src = torch.arange(atom_num).unsqueeze(-1).repeat(1, lattice_list.size(0))
    shift_src = torch.cat([shift_src[i,:] for i in range(shift_src.size(0))])
    
    indices = torch.where(mask)
    dist_target = dist[indices]
    u, v, _ = indices
    if use_lattice:
        u = torch.cat((u, shift_src), dim=0)
        v = torch.cat((v, shift_src), dim=0)
        dist_target = torch.cat((dist_target, shift), dim=0)
        assert u.size(0) == dist_target.size(0)

    return u, v, dist_target

def build_undirected_edgedata(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())

    return u, v, r


class PygGraph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0, 
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = False,
        use_angle: bool = False,
        include_coor = True
    ):
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r = build_undirected_edgedata(atoms, edges)
        elif neighbor_strategy == "pairwise-k-nearest":
            u, v, r = pair_nearest_neighbor_edges(
                atoms=atoms,
                pair_wise_distances=2,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        

        # build up atom attribute tensor
        sps_features = []
        for ii, (s, c) in enumerate(zip(atoms.elements, atoms.frac_coords)):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            if include_coor:
                feat.extend(c)
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
        g = Data(x=node_features, edge_index=edge_index, edge_attr=r)

        if compute_line_graph:
            linegraph_trans = LineGraph(force_directed=True)
            g_new = Data()
            g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
            lg = linegraph_trans(g)
            lg.edge_attr = pyg_compute_bond_cosines(lg)
            return g_new, lg
        else:
            return g
        
    @staticmethod
    def atom_dgl_multigraph_dimenet(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0, 
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = False,
        use_angle: bool = False,
        include_coor = True
    ):
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r = build_undirected_edgedata(atoms, edges)
        elif neighbor_strategy == "pairwise-k-nearest":
            u, v, r = pair_nearest_neighbor_edges(
                atoms=atoms,
                pair_wise_distances=2,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        

        # build up atom attribute tensor
        sps_features = []
        atomic_number_features = []
        coor_features = []

        for ii, (s, c) in enumerate(zip(atoms.elements, atoms.frac_coords)):
            atomic_number_feat = list(get_node_attributes(s, atom_features=atom_features))
            atomic_number_features.append(atomic_number_feat)
            coor_features.append(c)

        atomic_number_features = torch.tensor(np.squeeze(np.array(atomic_number_features))).type(
            torch.int
        )
        coor_features = torch.tensor(np.array(coor_features)).type(
            torch.get_default_dtype()
        )

        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()

        g = Data(z=atomic_number_features, pos=coor_features, edge_index=edge_index, edge_attr=r)

        if compute_line_graph:
            linegraph_trans = LineGraph(force_directed=True)
            g_new = Data()
            g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
            lg = linegraph_trans(g)
            lg.edge_attr = pyg_compute_bond_cosines(lg)
            return g_new, lg
        else:
            return g

def pyg_compute_bond_cosines(lg):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def pyg_compute_bond_angle(lg):
    """Compute bond angle from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    a = (r1 * r2).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(r1, r2).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle



class PygStandardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: Data):
        """Apply standardization to atom_features."""
        h = g.x
        g.x = (h - self.mean) / self.std
        return g



def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


def prepare_pyg_line_graph_batch(
    batch: Tuple[Tuple[Data, Data, torch.Tensor], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, lattice, t = batch
    batch = (
        (
            g.to(device),
            lg.to(device),
            lattice.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch

