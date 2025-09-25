from ase import Atoms 
from orb_models.forcefield import pretrained, atomic_system 




def orb_model_initializer(device = "cpu"):
    model = pretrained.orb_v3_conservative_omol(
        device = device, 
        precision = "float32-highest"
    )
    return model 


def predict_energy(
        model, 
        element_symbols, 
        xyz_coordinates, 
        charge = 0, 
        spin_multiplicity = 1,
        device = "cpu"
):
    """
    - Returns total energy prediction with units of kcal/mol 
    - 'element_symbols' is an array of element symbols for a single molecule 
    - 'xyz_coordinates' is an array containing arrays of nuclear coordinates
      corresponding to 'element_symbols' in units of angstrom
    """

    atoms = Atoms(
        symbols = element_symbols, 
        positions = [tuple(float(x) for x in coord) for coord in xyz_coordinates]
    )
    atoms.info["charge"] = charge 
    atoms.info["spin"] = spin_multiplicity
    graph = atomic_system.ase_atoms_to_atom_graphs(
        atoms, 
        model.system_config, 
        device = device
    )
    energy = float(
        model.predict(graph, split = False)["energy"].item()
    ) * 23.0609
    return energy 