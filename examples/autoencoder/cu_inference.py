import logging
import sys
sys.path.append('../')
from ase.io import Trajectory
from mlchem import Potentials
from mlchem.fingerprints import Gaussian
from mlchem.models.neuralnetwork import NeuralNetwork

logging.basicConfig(filename = 'cu_inference.log', level=logging.INFO,
                    format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')
# Load the images with ASE
images = Trajectory('cu_training.traj')


calc = Potentials.load(model='cu_training.mlchem', params='cu_training.params',
                       scaler='cu_training.scaler')

for atoms in images:
    energy = calc.get_potential_energy(atoms)
    print('mlchem predicted energy = {}' .format(energy))
    print('             DFT energy = {}' .format(atoms.get_potential_energy()))
