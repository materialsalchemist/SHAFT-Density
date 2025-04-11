from metrics.eval_metrics import *
from common.bonds_dictionary import bonds_dictionary
import time

def get_bs_dm_min_max(m: Structure, vpen_min=0.01,vpen_max=0.1,vpen_minmax=0.001):
    bond_min_atm = bonds_dictionary['min']
    bond_avg_atom = bonds_dictionary['mean']
    bond_std_atom = bonds_dictionary['std']

    a = m.atomic_numbers
    dist = m.distance_matrix
    nbond = 0
    vmin = 0
    vmax = 0 
    for i in range(0,len(a)-1):
        has_near_neighbour = False 
        for j in range(i+1,len(a)):
            nbond +=1
            bond_name = str(a[i]) + '_' + str(a[j])
            if bond_name not in bond_min_atm:
                bond_name = str(a[j]) + '_' + str(a[i])
                if bond_name not in bond_min_atm:
                    min_constraint = 1.0
                    neighbour_constraint = 3.5
                    # there is no bond dist data in the dictionary, use default value
                else:
                    min_constraint = bond_min_atm[bond_name]
                    neighbour_constraint = bond_avg_atom[bond_name] + bond_std_atom[bond_name]
            else:
                min_constraint = bond_min_atm[bond_name]
                # neighbour_constraint = bond_avg_atom[bond_name] + bond_std_atom[bond_name]
                neighbour_constraint = bond_avg_atom[bond_name] 

            if dist[i][j] < min_constraint:
                vmin+=1
            if has_near_neighbour == False:
                if dist[i][j] < neighbour_constraint:
                    has_near_neighbour = True
        if has_near_neighbour == False:
            vmax+= 1
    if nbond == 0:
        return vpen_minmax
    vmin = (nbond-vmin)/nbond
    vmax = (len(a) - vmax)/len(a)
    if vmin < 1 and vmax < 1:
        return vpen_minmax
    if vmin < 1:
        return vmin*vpen_min
    if vmax < 1:
        return vmax*vpen_max
    return 1.0


def get_pref_distance_matrix(m: Structure):
    bond_min_atm = bonds_dictionary['min']
    bond_avg_atom = bonds_dictionary['mean']

    a = m.atomic_numbers

    pref_dist_matrix = np.zeros((len(a),len(a)))

    for i in range(len(a)-1):
        for j in range(i+1, len(a)):
            bond_name = str(a[i]) + '_' + str(a[j])
            if bond_name not in bond_min_atm:
                bond_name = str(a[j]) + '_' + str(a[i])
            pref_dist_matrix[i,j] = pref_dist_matrix[j,i] = bond_avg_atom[bond_name]

    return pref_dist_matrix

def get_bs_pref_distm(m: Structure):
    dist = m.distance_matrix
    pref_dist = get_pref_distance_matrix(m)
    dis_dif = np.abs(dist-pref_dist)
    return np.mean(dis_dif)

def bond_distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)


def get_bond_data(structure: Structure):
    bond_data = {}
    for c1 in structure:
        c = structure.get_neighbors(site=c1, r=4)
        for c2 in c:
            Rij = bond_distance(c1.coords, c2.coords)

            if c1.specie.number < c2.specie.number:
                k = str(c1.specie.number)+'_'+str(c2.specie.number)
            else:
                k = str(c2.specie.number)+'_'+str(c1.specie.number)
            if k in bond_data.keys():
                available = False
                for bb in bond_data[k]:
                    if abs(bb-Rij) < 1e-5:
                        available = True
                        break
                if not available:
                    bond_data[k] += [Rij]
            else:
                bond_data[k] = [Rij]
    return bond_data


def get_bs_from_bond_data(m: Structure, vpen_min=0.01,vpen_max=0.1,vpen_minmax=0.001):
    bond_avg_atom = bonds_dictionary['mean']
    bond_min_atom = bonds_dictionary['min']
    
    bond_data = get_bond_data(m)
    if len(bond_data.keys())==0:
        return 5.0
    bs = 0
    for bk in bond_data.keys():
        bondlist = np.array(bond_data[bk])
        try:
            bond_min_dif = np.exp(-bondlist + bond_min_atom[bk])
        except:
            # If there is no info of that bond, use default value
            bond_min_atom = 1.0
            bond_min_dif = np.exp(-bondlist + 1.0)
        try:
            bond_dif = bondlist - bond_avg_atom[bk] 
        except:
            # If there is no info of that bond, use default value
            bond_avg_atom = 2.0
            bond_dif = bondlist - bond_avg_atom
        bond_dist_dif = np.mean(np.abs(bond_dif)+ bond_min_dif**2)
        bs += bond_dist_dif
    return bs/len(bond_data.keys())


def get_bs_min(m: Structure, vpen_min=0.01,vpen_max=0.1,vpen_minmax=0.001):
    a = m.atomic_numbers
    dist = m.distance_matrix
    nbond = 0
    vmin = 0
    min_constraint = 0.5
    for i in range(0,len(a)-1):
        for j in range(i+1,len(a)):
            nbond +=1
            if dist[i][j] < min_constraint:
                vmin+=1
    vmin = (nbond-vmin)/nbond
    if vmin < 1:
        return vmin*vpen_min
    return 1.0

def get_valid_score(structure):
    atom_types = [s.specie.Z for s in structure]
    elems, comps = get_composition(atom_types)
    return max(float(smact_validity(elems, comps)),0.0)


def get_merge_score(structure):
    ele_before_merge = len(np.unique(np.array(structure.atoms)))
    ele_after_merge = len(np.unique(np.array([s.species for s in structure.structure]).flatten()))
    # print('ele_before_merge',ele_before_merge)
    # print(np.unique(np.array(structure.atoms)))
    # print('ele_after_merge',ele_after_merge)
    # print(np.unique(np.array(s.species for s in structure.structure)))
    if ele_after_merge < ele_before_merge:
        return 0
    else:
        return 1


def reward_pref_bond_dict(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    pred = proxy(states)
    forme_norm = 10
    forme_score = (-pred)/forme_norm
    forme_score = np.exp(forme_score)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 1
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    bond_score = np.exp(-bond_score)
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    wes = 0.2
    wbs = 0.5
    wds = 0.2
    wvs = 0.1
    reward = wes*forme_score+ wds*density_score + wbs*bond_score + wvs*valid_score
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, forme_score, bond_score, valid_score, density_score, 0


def reward_pref_bond_dict_merge(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    pred = proxy(states)
    forme_norm = 10
    forme_score = (-pred)/forme_norm
    forme_score = np.exp(forme_score)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 1
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    bond_score = np.exp(-bond_score)
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    merge_score = np.array([get_merge_score(state) for state in states])
    wes = 0.2
    wbs = 0.5
    wds = 0.2
    wvs = 0.1
    wm = 0.2
    reward = wes*forme_score+ wds*density_score + wbs*bond_score + wvs*valid_score + wm*merge_score
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, forme_score, bond_score, valid_score, density_score, 0

def reward_only_bs(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    bond_score = np.exp(-bond_score)
    return 0, 0, bond_score, 0, 0 

def reward_metal_cluster(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    pred = proxy(states)
    forme_norm = 10
    forme_score = (-pred)/forme_norm
    forme_score = np.exp(forme_score)
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    bond_score = np.exp(-bond_score)
    merge_score = np.array([get_merge_score(state) for state in states])
    wes = 0.4
    wbs = 0.5
    wm = 0.1
    reward = wes*forme_score+ wbs*bond_score + wm*merge_score
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, forme_score, bond_score, 0, 0, 0

reward_functions_dict = {'reward_pref_bond_dict':reward_pref_bond_dict,
                         'reward_pref_bond_dict_merge':reward_pref_bond_dict_merge,
                         'reward_only_bs':reward_only_bs,
                         'reward_metal_cluster':reward_metal_cluster}
