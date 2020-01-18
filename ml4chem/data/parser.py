import json
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.io import Trajectory


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
        molecule.set_calculator(SinglePointCalculator())
        molecule.calc.results["energy"] = energy
        atoms.append(molecule)

        if trajfile is not None:
            traj.write(molecule, energy=energy)

    return atoms


class SinglePointCalculator(Calculator):
    """A SinglePointCalculator class

    This class creates a fake calculator that is used to populate
    calc.results dictionaries in ASE objects.

    Parameters
    ----------
    implemented_properties : list
        List with supported properties.
    """

    def __init__(self, implemented_properties=None):
        super(SinglePointCalculator, self).__init__()
        if implemented_properties is None:
            self.implemented_properties = ["energy", "forces"]

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

    @staticmethod
    def get_forces(atoms):
        """Get atomic forces

        Parameters
        ----------
        atoms : obj
            Atoms objects

        Returns
        -------
        forces
            The atomic force of the molecule.
        """
        return atoms.calc.results["forces"]


def ani_to_ase(hdf5file, data_keys, trajfile=None):
    """ANI to ASE

    Parameters
    ----------
    hdf5file : hdf5, list
        hdf5 file loaded using pyanitools (or list of them).
    data_keys : list
        List of keys to extract data.
    trajfile : str, optional
        Name of trajectory file to be saved, by default None.

    Returns
    -------
    atoms
        A list of Atoms objects.
    """

    if isinstance(hdf5file, list) is False:
        hdf5file = [hdf5file]

    atoms = []
    prop = {"energies": "energy", "energy": "energy"}

    if trajfile is not None:
        traj = Trajectory(trajfile, mode="w")

    for hdf5 in hdf5file:
        for data in hdf5:

            symbols = data["species"]
            conformers = data["coordinates"]

            for index, conformer in enumerate(conformers):
                molecule = Atoms(positions=conformer, symbols=symbols)
                molecule.set_calculator(SinglePointCalculator())

                _prop = {}

                for key in data_keys:
                    value = data[key][index]

                    # Mutate key because ANI naming is not standard.
                    key = prop[key]
                    _prop[key] = value

                    molecule.calc.results[key] = value

                atoms.append(molecule)

                if trajfile is not None:
                    traj.write(molecule, **_prop)

    return atoms
