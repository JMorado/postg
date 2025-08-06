# Using external weights in postg

This fork has been modified to accept external atomic weights (such as MBIS weights) instead of using the built-in Hirshfeld weights. 

## Getting the mesh 

To use external weights, you first need to obtain the exact mesh points that postg generates. Use the `writemesh` keyword:

```bash
postg 0.6356 1.5119 molecule.wfx b3lyp writemesh mesh_points.dat
```

This will write all mesh points and their integration weights to `mesh_points.dat` in the format:

```
# Mesh points for postg
# Number of points: 12345
# Format: x y z weight
  1.234567890123456E+00   2.345678901234567E+00   3.456789012345678E+00   4.567890123456789E-03
  ...
```

You can then use this mesh to compute atomic weights from your partitioned density.

## Usage with external weights

Once you have computed your weights (e.g., by running `mbis_weights.py`), add the `weights` keyword followed by a filename to your postg command line:

```bash
postg 0.6356 1.5119 orca.wfx b3lyp weights mbis_weights.dat
```

## Weight File Format

The weight file should contain one line per mesh point with the weight for each atom separated by spaces:

```
weight_atom1_point1  weight_atom2_point1  weight_atom3_point1  ...
weight_atom1_point2  weight_atom2_point2  weight_atom3_point2  ...
weight_atom1_point3  weight_atom2_point3  weight_atom3_point3  ...
...
```

### Requirements

1. Number of lines: Must equal the number of mesh points (printed in postg output as "mesh size")
2. Number of columns: Must equal the number of atoms in the molecule 
3. Weight values: Should be between 0 and 1 for each atom at each grid point
4. Normalization: Weights for all atoms at each grid point should sum to 1 (or close to 1)

### Example for a 3-atom molecule with 5 mesh points:

```
0.8 0.15 0.05
0.2 0.7 0.1  
0.1 0.1 0.8
0.6 0.3 0.1
0.05 0.05 0.9
```

### Utility Scripts

The repository includes helper scripts:

- **`mesh_utils.py`**: Convert mesh formats and validate weight files
  ```bash
  # Convert mesh for use with multiwfn
  python mesh_utils.py convert mesh_points.dat mesh_for_multiwfn.xyz
  
  # Validate your weights file
  python mesh_utils.py validate mbis_weights.dat 12345 3
  
  # Show mesh info
  python mesh_utils.py info mesh_points.dat
  ```

## Notes

- The mesh generation in postg is deterministic and depends on the atomic positions and atomic numbers.
- Ensure that your external weight calculation is performed using exactly the same mesh points.
- The order of the mesh points follows postg's internal atom-centered mesh generation scheme.
