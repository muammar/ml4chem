import sys
from dask.distributed import Client, LocalCluster
sys.path.append('../../')
from ase.io import Trajectory
from mlchem import Potentials

def main():
    """docstring for main"""

    # Load the images with ASE
    images = Trajectory('cu_training.traj')

    calc = Potentials.load(model='cu_training.mlchem', params='cu_training.params',
                           scaler='cu_training.scaler')

    for atoms in images:
        energy = calc.get_potential_energy(atoms)
        print('mlchem predicted energy = {}' .format(energy))
        print('             DFT energy = {}' .format(atoms.get_potential_energy()))

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=8, threads_per_worker=2)
    client = Client(cluster, asyncronous=True)
    main()
