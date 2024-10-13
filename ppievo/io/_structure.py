import numpy as np
from typing import List
from biotite.structure.io import pdb
import biotite.structure as struc
from biotite.sequence import ProteinSequence


""" Functions to process structures """


def load_structure(fpath, chain=None):
    """
    Load pdb structure as biotite.structure
    Filter to canonical residues
    Filter out residues with incomplete backbone atoms

    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """

    def _filter_incomplete_bb(res, axis=None):
        bb_mask = (
            (res.atom_name == "N")
            | (res.atom_name == "CA")
            | (res.atom_name == "C")
            | (res.atom_name == "O")
        )
        return bb_mask.sum() == 4

    with open(fpath) as fin:
        pdbf = pdb.PDBFile.read(fin)
    structure = pdb.get_structure(pdbf, model=1)
    std_aa_mask = struc.filter_canonical_amino_acids(structure)
    structure = structure[std_aa_mask]
    bb_mask = struc.apply_residue_wise(structure, structure, _filter_incomplete_bb)
    bb_mask = struc.spread_residue_wise(structure, bb_mask)
    structure = structure[bb_mask]

    all_chains = struc.get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    structure = structure[structure.element != "H"]
    return structure


def get_atom_coords_residuewise(
    struct: struc.AtomArray, atoms: List[str] = ["N", "CA", "C"]
):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """

    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return struc.apply_residue_wise(struct, struct, filterfn)


def extract_coords_from_structure(
    structure: struc.AtomArray,
    atoms: List[str] = ["N", "CA", "C"],
    as_list: bool = False,
):
    """
    Args:
        atoms: List of atom names to extract coordinates
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x K x 3 array for the coordinates of K atoms
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(structure, atoms)
    if as_list:
        coords = coords.tolist()
    residue_identities = struc.get_residues(structure)[1]
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq


def extract_coords_from_complex(
    structure: struc.AtomArray,
    atoms: List[str] = ["N", "CA", "C"],
    as_list: bool = False,
):
    """
    Args:
        structure: biotite AtomArray
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x K x 3 array
          coordinates representing the K atoms of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    coords = {}
    seqs = {}
    all_chains = struc.get_chains(structure)
    for chain_id in all_chains:
        chain = structure[structure.chain_id == chain_id]
        coord, seq = extract_coords_from_structure(chain, atoms, as_list)
        seqs[chain_id] = seq
        coords[chain_id] = coord
    return coords, seqs


def extract_resids_from_complex(structure: struc.AtomArray, as_list: bool = False):
    """
    Args:
        structure: biotite AtomArray
    Returns:
        resids: Dictionary mapping chain ids to np.Array of res ids
            of each chain
    """
    resids = {}
    all_chains = struc.get_chains(structure)
    for chain_id in all_chains:
        chain = structure[structure.chain_id == chain_id]
        res_id = struc.get_residues(chain)[0]
        if as_list:
            res_id = res_id.tolist()
        resids[chain_id] = res_id
    return resids
