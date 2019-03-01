import sys
sys.path.append('../')
from ase.io import Trajectory
from mlchem import Potentials
from mlchem.fingerprints import Gaussian
from mlchem.models.neuralnetwork import NeuralNetwork

# Load the images with ASE
images = Trajectory('cu_training.traj')


calc = Potentials.load(model='cu_training.mlchem', params='cu_training.params',
                       scaler='cu_training.scaler')

for atoms in images:
    energy = calc.get_potential_energy(atoms)
    print('mlchem predicted energy = {}' .format(energy))
