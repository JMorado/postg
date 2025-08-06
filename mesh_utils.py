#!/usr/bin/env python3
"""
Convert postg mesh file to format suitable for multiwfn or other programs.

This module provides utilities for working with postg mesh and weight files,
including format conversion and validation.
"""

import numpy as np
import sys
import argparse
from typing import Tuple, Optional
import numpy.typing as npt


def read_postg_mesh(filename: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Read mesh points and weights from postg output file.

    Parameters
    ----------
    filename : str
        Path to the postg mesh output file.

    Returns
    -------
    points : np.ndarray
        Array of shape (n_points, 3) containing mesh point coordinates.
    weights : np.ndarray
        Array of shape (n_points,) containing mesh point weights.

    Notes
    -----
    The postg mesh file should contain lines with at least 4 columns:
    x, y, z coordinates and weight. Lines starting with '#' are treated
    as comments and ignored.
    """
    points = []
    weights = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                x, y, z, w = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
                weights.append(w)
    
    return np.array(points), np.array(weights)


def write_xyz_mesh(points: npt.NDArray[np.float64], filename: str) -> None:
    """
    Write mesh points in simple XYZ format for multiwfn.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_points, 3) containing mesh point coordinates.
    filename : str
        Output filename for the XYZ format file.

    Notes
    -----
    The XYZ format includes a header with the number of points and
    a comment line, followed by one line per point with dummy atom
    type 'X' and coordinates.
    """
    with open(filename, 'w') as f:
        f.write(f"{len(points)}\n")
        f.write("Mesh points for multiwfn\n")
        for i, point in enumerate(points):
            f.write(f"X {point[0]:.15f} {point[1]:.15f} {point[2]:.15f}\n")


def validate_weights(weights_file: str, expected_npoints: int, expected_natoms: int) -> bool:
    """
    Validate a weights file format.

    Parameters
    ----------
    weights_file : str
        Path to the weights file to validate.
    expected_npoints : int
        Expected number of mesh points.
    expected_natoms : int
        Expected number of atoms.

    Returns
    -------
    bool
        True if validation passes, False otherwise.

    Notes
    -----
    Validation checks include:
    - Correct array shape (n_points x n_atoms)
    - Proper normalization (each row sums to ~1.0)
    - Non-negativity of weights
    """
    try:
        weights = np.loadtxt(weights_file)
        
        if weights.shape != (expected_npoints, expected_natoms):
            print(f"ERROR: Expected shape ({expected_npoints}, {expected_natoms}), got {weights.shape}")
            return False
        
        # Check normalization
        row_sums = np.sum(weights, axis=1)
        min_sum, max_sum = np.min(row_sums), np.max(row_sums)
        
        print(f"Weights validation:")
        print(f"  Shape: {weights.shape} ✓")
        print(f"  Sum range: [{min_sum:.6f}, {max_sum:.6f}]")
        
        if abs(min_sum - 1.0) > 0.01 or abs(max_sum - 1.0) > 0.01:
            print(f"  WARNING: Weights are not well normalized (should sum to 1.0)")
        else:
            print(f"  Normalization: ✓")
        
        # Check for negative weights
        negative_count = np.sum(weights < 0)
        if negative_count > 0:
            print(f"  WARNING: {negative_count} negative weights found")
        else:
            print(f"  Non-negativity: ✓")
            
        return True
        
    except Exception as e:
        print(f"ERROR reading weights file: {e}")
        return False


def main() -> None:
    """
    Main function for command-line interface.

    Provides subcommands for:
    - convert: Convert postg mesh to XYZ format
    - validate: Validate weights file format
    - info: Show mesh file information
    """
    parser = argparse.ArgumentParser(description='Postg mesh and weights utilities')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert mesh command
    convert_parser = subparsers.add_parser('convert', help='Convert postg mesh to XYZ format')
    convert_parser.add_argument('input', help='Input mesh file from postg')
    convert_parser.add_argument('output', help='Output XYZ file for multiwfn')
    
    # Validate weights command
    validate_parser = subparsers.add_parser('validate', help='Validate weights file')
    validate_parser.add_argument('weights', help='Weights file to validate')
    validate_parser.add_argument('npoints', type=int, help='Expected number of mesh points')
    validate_parser.add_argument('natoms', type=int, help='Expected number of atoms')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show mesh file info')
    info_parser.add_argument('mesh', help='Mesh file from postg')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        points, weights = read_postg_mesh(args.input)
        write_xyz_mesh(points, args.output)
        print(f"Converted {len(points)} mesh points from {args.input} to {args.output}")
        
    elif args.command == 'validate':
        validate_weights(args.weights, args.npoints, args.natoms)
        
    elif args.command == 'info':
        points, weights = read_postg_mesh(args.mesh)
        print(f"Mesh file: {args.mesh}")
        print(f"Number of points: {len(points)}")
        print(f"Weight range: [{np.min(weights):.6f}, {np.max(weights):.6f}]")
        print(f"Total weight: {np.sum(weights):.6f}")
        
        # Show coordinate ranges
        print("Coordinate ranges:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            print(f"  {axis}: [{np.min(points[:,i]):.3f}, {np.max(points[:,i]):.3f}]")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
