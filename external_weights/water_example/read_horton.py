import h5py


with h5py.File('horton.out', 'r') as file:
    print(file.keys())
    print(file["charges"][0:10000])
    print(file["core_charges"][0:10000])
    print(file["valence_charges"][0:10000])
    print(file["valence_charges"][0:10000]+file["core_charges"][0:100000])








