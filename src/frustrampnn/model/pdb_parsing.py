"""
PDB parsing utilities for FrustraMPNN.

This module contains functions for parsing PDB files and extracting
structural information needed for ProteinMPNN.

Original source: ProteinMPNN by Dauparas et al. (2022)
https://github.com/dauparas/ProteinMPNN
"""


import numpy as np

__all__ = [
    "parse_PDB",
    "parse_PDB_biounits",
    "alt_parse_PDB",
    "alt_parse_PDB_biounits",
]


def parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    """Parse a PDB file and extract coordinates for specified atoms.

    Args:
        x: PDB filename
        atoms: List of atom names to extract (default: ['N', 'CA', 'C'])
        chain: Chain ID to extract (default: None for all chains)

    Returns:
        tuple: (coordinates array of shape (length, atoms, 3), sequence list)
    """
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

    def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x)
        if x.ndim == 0:
            x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x)
        if x.ndim == 1:
            x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12 + 4].strip()
                resi = line[17:17 + 3]
                resn = line[22:22 + 5].strip()
                x_coord, y_coord, z_coord = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x_coord, y_coord, z_coord])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except TypeError:
        return 'no_chain', 'no_chain'


def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False, side_chains=False):
    """Parse a PDB file and return a list of dictionaries with structural data.

    Args:
        path_to_pdb: Path to PDB file
        input_chain_list: List of chain IDs to extract (default: None for all)
        ca_only: If True, only extract CA atoms
        side_chains: If True, extract all sidechain atoms

    Returns:
        list: List of dictionaries containing sequence and coordinate data
    """
    c = 0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            elif side_chains:
                sidechain_atoms = ["N", "CA", "C", "O", "CB",
                                   "CG", "CG1", "OG1", "OG2", "CG2", "OG", "SG",
                                   "CD", "SD", "CD1", "ND1", "CD2", "OD1", "OD2", "ND2",
                                   "CE", "CE1", "NE1", "OE1", "NE2", "OE2", "NE", "CE2", "CE3",
                                   "NZ", "CZ", "CZ2", "CZ3", "CH2", "OH", "NH1", "NH2"]
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


def alt_parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    """Alternative PDB parser that also returns residue number list.

    Args:
        x: PDB filename
        atoms: List of atom names to extract
        chain: Chain ID to extract

    Returns:
        tuple: (coordinates, sequence, residue_number_list)
    """
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

    def AA_to_N(x):
        x = np.array(x)
        if x.ndim == 0:
            x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        x = np.array(x)
        if x.ndim == 1:
            x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    resn_list = []
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12 + 4].strip()
                resi = line[17:17 + 3]
                resn = line[22:22 + 5].strip()
                # RAW resn is defined HERE
                if resn not in resn_list:
                    resn_list.append(resn)  # NEED to keep ins code here
                x_coord, y_coord, z_coord = [float(line[i:(i + 8)]) for i in [30, 38, 46]]
                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x_coord, y_coord, z_coord])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)), list(dict.fromkeys(resn_list))
    except TypeError:
        return 'no_chain', 'no_chain', 'no_chain'


def alt_parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False, side_chains=False):
    """Alternative PDB parser that includes residue number list.

    Args:
        path_to_pdb: Path to PDB file
        input_chain_list: List of chain IDs to extract
        ca_only: If True, only extract CA atoms
        side_chains: If True, extract all sidechain atoms

    Returns:
        list: List of dictionaries with sequence, coordinates, and residue numbers
    """
    c = 0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        my_dict['resn_list'] = []
        s = 0
        concat_seq = ''
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            elif side_chains:
                sidechain_atoms = ["N", "CA", "C", "O", "CB",
                                   "CG", "CG1", "OG1", "OG2", "CG2", "OG", "SG",
                                   "CD", "SD", "CD1", "ND1", "CD2", "OD1", "OD2", "ND2",
                                   "CE", "CE1", "NE1", "OE1", "NE2", "OE2", "NE", "CE2", "CE3",
                                   "NZ", "CZ", "CZ2", "CZ3", "CH2", "OH", "NH1", "NH2"]
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq, resn_list = alt_parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)

            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq

        # old handling for resn lists
        my_dict['resn_list'] = list(resn_list)
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


