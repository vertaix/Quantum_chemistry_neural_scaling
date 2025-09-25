from ase import Atoms 
from ase.build import molecule 
from mendeleev import element 
from fairchem.core import pretrained_mlip, FAIRChemCalculator 




def uma_s_1p1_model_initializer(
        device = "cpu"
):
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device = device)
    calc = FAIRChemCalculator(predictor, task_name = "omol")
    return calc 


def uma_m_1p1_model_initializer(
        device = "cpu"
):
    predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device = device)
    calc = FAIRChemCalculator(predictor, task_name = "omol")
    return calc 


def omol25_esen_sm_conserving_model_initializer(
        model_path,
        device = "cpu"
):
    predictor = pretrained_mlip.load_predict_unit(
        path = model_path, 
        device = device 
    )
    calc = FAIRChemCalculator(predictor, task_name = "omol")
    return calc 


def predict_energy(
        model, 
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

    atoms = Atoms(
        symbols = element_symbols, 
        positions = [tuple(float(x) for x in coord) for coord in xyz_coordinates]
    )
    atoms.info.update({"spin": spin_multiplicity, "charge": charge})
    atoms.calc = model 
    energy = atoms.get_total_energy() * 23.0609 
    return energy 