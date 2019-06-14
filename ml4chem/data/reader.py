import json
from ase import Atom, Atoms
from ase.io import Trajectory, write


# to get the total energy
def mol_total_energy(cjson):
    total_energy = cjson["cjson"]["properties"]["energy"]["total"]
    return total_energy


def cjson_to_ase(cjson):
    coord_start = 0
    coord_end = 3
    energy = mol_total_energy(cjson)
    atomic_numbers = cjson["cjson"]["atoms"]["elements"]["number"]
    positions = cjson["cjson"]["atoms"]["coords"]["3d"]
    atoms = []
    for number in atomic_numbers:
        mol_positions = positions
        atom = Atom(number, position=mol_positions[coord_start:coord_end])
        atoms.append(atom)
        coord_start += 3
        coord_end += 3
    return Atoms(atoms)


def cjson_reader(cjsonfile, trajfile):
    collection = json.loads(open(cjsonfile, 'r').read())
    traj = Trajectory(trajfile, mode='w')
    for document in collection:
        cjson = json.loads(document)
        traj.write(cjson_to_ase(cjson), energy=mol_total_energy(cjson))
	#need to add return later