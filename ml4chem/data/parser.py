import json
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.io import Trajectory, write


# to get the total energy
def get_total_energy(cjson):
    total_energy = cjson["cjson"]["properties"]["energy"]["total"]
    return total_energy


def cjson_to_ase(cjson):
    coord_start = 0
    coord_end = 3
    energy = get_total_energy(cjson)
    atomic_numbers = cjson["cjson"]["atoms"]["elements"]["number"]
    positions = cjson["cjson"]["atoms"]["coords"]["3d"]
    atoms = []
    for number in atomic_numbers:
        mol_positions = positions
        atom = Atom(number, position=mol_positions[coord_start:coord_end])
        atoms.append(atom)
        coord_start += 3
        coord_end += 3
    return Atoms(atoms), energy


def cjson_parser(cjsonfile, trajfile=None):
    """Parse CJSON files

    Parameters
    ----------
    cjsonfile : str
        Path to the CJSON file.
    trajfile : str, optional
        Name of trajectory file to be saved, by default None.

    Returns
    -------
    atoms
        A list of Atoms objects.
    """

    collection = json.loads(open(cjsonfile, "r").read())

    atoms = []

    if trajfile is not None:
        traj = Trajectory(trajfile, mode="w")

    for document in collection:
        cjson = json.loads(document)
        molecule, energy = cjson_to_ase(cjson)
        molecule.set_calculator(FakeCalculator())
        molecule.calc.results["energy"] = energy
        atoms.append(molecule)

        if trajfile is not None:
            traj.write(molecule, energy=energy)

    return atoms


class FakeCalculator(Calculator):
    """A FakeCalculator class

    This class creates a fake calculator that is used to populate
    calc.results dictionaries in ASE objects. 

    Parameters
    ----------
    implemented_properties : list
        List with supported properties.
    """

    def __init__(self, implemented_properties=None):
        super(FakeCalculator, self).__init__()
        if implemented_properties is None:
            self.implemented_properties = ["energy"]

    @staticmethod
    def get_potential_energy(atoms):
        """Get the potential energy

        Parameters
        ----------
        atoms : obj
            Atoms objects

        Returns
        -------
        energy
            The energy of the molecule.
        """
        return atoms.calc.results["energy"]
