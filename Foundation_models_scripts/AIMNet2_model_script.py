from ase import Atoms 
from aimnet.calculators import AIMNet2ASE 




def predict_energy(
        element_symbols, 
        xyz_coordinates, 
        charge = 0, 
        spin_multiplicity = 1
):
    """
    - Returns total energy prediction with units of kcal/mol 
    - 'element_symbols' is an array of element symbols for a single molecule 
    - 'xyz_coordinates' is an array containing arrays of nuclear coordinates
      corresponding to 'element_symbols' in units of angstrom
    """
    calc = AIMNet2ASE("aimnet2)
    calc.set_charge(int(charge))
    atoms = Atoms(
        symbols = element_symbols, 
        positions = [tuple(float(x) for x in coord) for coord in xyz_coordinates]
    )
    atoms.info["spin"] = int(spin_multiplicity)
    energy = float(atoms.get_potential_energy()) * 23.0609
    return energy 
