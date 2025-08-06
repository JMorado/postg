#!/usr/bin/env python3
"""
MBIS density weight computation for postg.

This module computes MBIS (Minimal Basis Iterative Stockholder) weights
for mesh points using valence charges and widths from Horton calculations.
"""

import numpy as np
import h5py
import sys
import re
from typing import Tuple, List, Union
import numpy.typing as npt

def load_mesh_points(mesh_filename: str) -> npt.NDArray[np.float64]:
    """
    Load mesh points from a file.

    Parameters
    ----------
    mesh_filename : str
        Path to the mesh file containing point coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) containing mesh point coordinates.
    
    Notes
    -----
    The mesh file should contain lines with at least 4 columns:
    x, y, z coordinates and optionally other data. Lines starting
    with '#' are treated as comments and ignored.
    """
    points = []
    with open(mesh_filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[:3])
            points.append(np.array([x, y, z]))
    return np.array(points)


def read_positions_from_wfx(wfx_file: str) -> npt.NDArray[np.float64]:
    """
    Read atomic positions from a WFX file.

    Parameters
    ----------
    wfx_file : str
        Path to the WFX file.

    Returns
    -------
    np.ndarray
        Array of shape (n_atoms, 3) containing atomic coordinates.

    Raises
    ------
    RuntimeError
        If the Nuclear Cartesian Coordinates section is not found.
    """
    positions = []
    with open(wfx_file, 'r') as f:
        content = f.read()
    coords_match = re.search(r'<Nuclear Cartesian Coordinates>(.*?)</Nuclear Cartesian Coordinates>', content, re.DOTALL)
    if not coords_match:
        raise RuntimeError("Could not find <Nuclear Cartesian Coordinates> section in WFX file")
    coords_block = coords_match.group(1)
    for line in coords_block.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        x, y, z = map(float, parts[:3])
        positions.append([x, y, z])
    return np.array(positions)


def read_positions_from_wfn(wfn_file: str) -> npt.NDArray[np.float64]:
    """
    Read atomic positions from a WFN file.

    Parameters
    ----------
    wfn_file : str
        Path to the WFN file.

    Returns
    -------
    np.ndarray
        Array of shape (n_atoms, 3) containing atomic coordinates.

    Raises
    ------
    RuntimeError
        If the atomic coordinates section is not found.
    """
    positions = []
    with open(wfn_file, 'r') as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if 'Center      Atomic' in line:
            start = i + 2
            break
    if start is None:
        raise RuntimeError("Could not find atomic coordinates section in WFN file")
    for line in lines[start:]:
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 5:
            break
        x, y, z = map(float, parts[2:5])
        positions.append([x, y, z])
    return np.array(positions)


def rho_valence(r: float, n_valence: float, valence_width: float) -> float:
    """
    Calculate the analytical MBIS valence density at distance r.

    Parameters
    ----------
    r : float
        Distance from the atom center in Bohr.
    n_valence : float
        Number of valence electrons.
    valence_width : float
        Valence width parameter (S).

    Returns
    -------
    float
        MBIS valence density at distance r.

    Notes
    -----
    The MBIS valence density is given by:
    rho(r) = (N * S^3)/(8 * pi) * exp(-S * r)
    where r is in Bohr, valence_width = S, valence_charge = N.
    """
    if r < 0:
        return 0.0
    coef = n_valence / (8 * np.pi * valence_width ** 3)
    return coef * np.exp(-r / valence_width)


def compute_weights(
    mesh_points: npt.NDArray[np.float64],
    atom_positions: npt.NDArray[np.float64],
    valence_charges: npt.NDArray[np.float64],
    valence_widths: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute MBIS weights for mesh points.

    Parameters
    ----------
    mesh_points : np.ndarray
        Array of shape (n_points, 3) containing mesh point coordinates.
    atom_positions : np.ndarray
        Array of shape (n_atoms, 3) containing atomic coordinates.
    valence_charges : np.ndarray
        Array of shape (n_atoms,) containing valence charges.
    valence_widths : np.ndarray
        Array of shape (n_atoms,) containing valence width parameters.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, n_atoms) containing MBIS weights.
        Each row sums to 1.0 (normalized weights).
    """
    weights = []
    for p in mesh_points:
        rho_vals = []
        for pos, N, S in zip(atom_positions, valence_charges, valence_widths):
            r = np.linalg.norm(p - pos)
            rho = rho_valence(r, -N, S)
            rho_vals.append(rho)
        total = sum(rho_vals)
        if total > 0:
            weights.append([v / total for v in rho_vals])
        else:
            # avoid division by zero, evenly distribute zero weights
            weights.append([0.0] * len(rho_vals))
    return np.array(weights)


def write_weights(filename: str, weights: npt.NDArray[np.float64]) -> None:
    """
    Write MBIS weights to a file.

    Parameters
    ----------
    filename : str
        Output filename for the weights.
    weights : np.ndarray
        Array of shape (n_points, n_atoms) containing MBIS weights.

    Notes
    -----
    The weights are written in plain text format with 10 decimal places
    of precision, one row per mesh point.
    """
    np.savetxt(filename, weights, fmt='%.10f')


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} mesh_points.dat horton_file.h5 molecule.wfx_or_wfn output_weights.dat")
        sys.exit(1)

    mesh_file = sys.argv[1]
    horton_file = sys.argv[2]
    mol_file = sys.argv[3]
    output_file = sys.argv[4]

    print(f"Loading mesh points from {mesh_file} ...")
    mesh_points = load_mesh_points(mesh_file)

    print(f"Reading atomic positions from {mol_file} ...")
    if mol_file.endswith('.wfx'):
        atom_positions = read_positions_from_wfx(mol_file)
    elif mol_file.endswith('.wfn'):
        atom_positions = read_positions_from_wfn(mol_file)
    else:
        print("ERROR: Only .wfx or .wfn files supported for atomic positions.")
        sys.exit(1)

    print(f"Loading MBIS valence charges and widths from {horton_file} ...")
    with h5py.File(horton_file, 'r') as f:
        if 'valence_charges' not in f or 'valence_widths' not in f:
            print("ERROR: valence_charges or valence_widths datasets not found in Horton .h5 file.")
            sys.exit(1)
        valence_charges = f['valence_charges'][:]
        valence_widths = f['valence_widths'][:]

    if len(atom_positions) != len(valence_charges):
        print(f"WARNING: Number of atoms in positions ({len(atom_positions)}) "
              f"does not match number of valence parameters ({len(valence_charges)})")

    print(f"Computing MBIS weights for {len(mesh_points)} points and {len(atom_positions)} atoms ...")
    weights = compute_weights(mesh_points, atom_positions, valence_charges, valence_widths)

    print(f"Writing weights to {output_file} ...")
    write_weights(output_file, weights)

    print("Done!")

