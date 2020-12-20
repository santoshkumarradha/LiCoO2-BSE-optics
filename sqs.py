import itertools
import random
from typing import List, Dict, Tuple, Union
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols as periodic_table


def generate_target_structure_from_supercells(
        cluster_space: ClusterSpace,
        supercells: List[Atoms],
        target_concentrations: dict,
        target_cluster_vector: List[float],
        T_start: float = 5.0,
        T_stop: float = 0.001,
        n_steps: int = None,
        optimality_weight: float = 1.0,
        random_seed: int = None,
        tol: float = 1e-5) -> Atoms:
    #--
    target_concentrations = _validate_concentrations(target_concentrations,
                                                     cluster_space)

    calculators = []

    valid_supercells = []
    warning_issued = False
    for supercell in supercells:
        supercell_copy = supercell.copy()
        try:
            occupy_structure_randomly(supercell_copy, cluster_space,
                                      target_concentrations)
        except ValueError:
            if not warning_issued:
                logger.warning(
                    'At least one supercell was not commensurate with the specified '
                    'target concentrations.')
                warning_issued = True
            continue
        valid_supercells.append(supercell_copy)
        calculators.append(
            TargetVectorCalculator(supercell_copy,
                                   cluster_space,
                                   target_cluster_vector,
                                   optimality_weight=optimality_weight,
                                   optimality_tol=tol))

    if len(valid_supercells) == 0:
        raise ValueError('No supercells that may host the specified '
                         'target_concentrations were supplied.')

    ens = TargetClusterVectorAnnealing(structure=valid_supercells,
                                       calculators=calculators,
                                       T_start=T_start,
                                       T_stop=T_stop,
                                       random_seed=random_seed)
    return ens.generate_structure(number_of_trial_steps=n_steps)


def generate_target_structure(cluster_space: ClusterSpace,
                              max_size: int,
                              target_concentrations: dict,
                              target_cluster_vector: List[float],
                              include_smaller_cells: bool = True,
                              pbc: Union[Tuple[bool, bool, bool],
                                         Tuple[int, int, int]] = None,
                              T_start: float = 5.0,
                              T_stop: float = 0.001,
                              n_steps: int = None,
                              optimality_weight: float = 1.0,
                              random_seed: int = None,
                              tol: float = 1e-5) -> Atoms:
    #--
    target_concentrations = _validate_concentrations(target_concentrations,
                                                     cluster_space)

    if pbc is None:
        pbc = (True, True, True)

    prim = cluster_space.primitive_structure
    prim.set_pbc(pbc)

    supercells = []
    if include_smaller_cells:
        sizes = list(range(1, max_size + 1))
    else:
        sizes = [max_size]

    for size in sizes:

        supercell = cluster_space.primitive_structure.repeat((size, 1, 1))
        if not _concentrations_fit_structure(
                structure=supercell,
                cluster_space=cluster_space,
                concentrations=target_concentrations):
            continue

        for supercell in enumerate_supercells(prim, [size]):
            supercell.set_pbc(True)
            supercells.append(supercell)

    return generate_target_structure_from_supercells(
        cluster_space=cluster_space,
        supercells=supercells,
        target_concentrations=target_concentrations,
        target_cluster_vector=target_cluster_vector,
        T_start=T_start,
        T_stop=T_stop,
        n_steps=n_steps,
        optimality_weight=optimality_weight,
        random_seed=random_seed,
        tol=tol)


def generate_sqs_from_supercells(cluster_space: ClusterSpace,
                                 supercells: List[Atoms],
                                 target_concentrations: dict,
                                 T_start: float = 5.0,
                                 T_stop: float = 0.001,
                                 n_steps: int = None,
                                 optimality_weight: float = 1.0,
                                 random_seed: int = None,
                                 tol: float = 1e-5) -> Atoms:
    #--

    sqs_vector = _get_sqs_cluster_vector(
        cluster_space=cluster_space,
        target_concentrations=target_concentrations)
    return generate_target_structure_from_supercells(
        cluster_space=cluster_space,
        supercells=supercells,
        target_concentrations=target_concentrations,
        target_cluster_vector=sqs_vector,
        T_start=T_start,
        T_stop=T_stop,
        n_steps=n_steps,
        optimality_weight=optimality_weight,
        random_seed=random_seed,
        tol=tol)


def generate_sqs(cluster_space: ClusterSpace,
                 max_size: int,
                 target_concentrations: dict,
                 include_smaller_cells: bool = True,
                 pbc: Union[Tuple[bool, bool, bool], Tuple[int, int,
                                                           int]] = None,
                 T_start: float = 5.0,
                 T_stop: float = 0.001,
                 n_steps: int = None,
                 optimality_weight: float = 1.0,
                 random_seed: int = None,
                 tol: float = 1e-5) -> Atoms:
    #--

    sqs_vector = _get_sqs_cluster_vector(
        cluster_space=cluster_space,
        target_concentrations=target_concentrations)
    return generate_target_structure(
        cluster_space=cluster_space,
        max_size=max_size,
        target_concentrations=target_concentrations,
        target_cluster_vector=sqs_vector,
        include_smaller_cells=include_smaller_cells,
        pbc=pbc,
        T_start=T_start,
        T_stop=T_stop,
        n_steps=n_steps,
        optimality_weight=optimality_weight,
        random_seed=random_seed,
        tol=tol)


def generate_sqs_by_enumeration(cluster_space: ClusterSpace,
                                max_size: int,
                                target_concentrations: dict,
                                include_smaller_cells: bool = True,
                                pbc: Union[Tuple[bool, bool, bool],
                                           Tuple[int, int, int]] = None,
                                optimality_weight: float = 1.0,
                                tol: float = 1e-5) -> Atoms:
    #--
    target_concentrations = _validate_concentrations(target_concentrations,
                                                     cluster_space)
    sqs_vector = _get_sqs_cluster_vector(
        cluster_space=cluster_space,
        target_concentrations=target_concentrations)

    cr = {}

    sublattices = cluster_space.get_sublattices(
        cluster_space.primitive_structure)
    for sl in sublattices:
        mult_factor = len(sl.indices) / len(cluster_space.primitive_structure)
        if sl.symbol in target_concentrations:
            sl_conc = target_concentrations[sl.symbol]
        else:
            sl_conc = {sl.chemical_symbols[0]: 1.0}
        for species, value in sl_conc.items():
            c = value * mult_factor
            if species in cr:
                cr[species] = (cr[species][0] + c, cr[species][1] + c)
            else:
                cr[species] = (c, c)

    c_sum = sum(c[0] for c in cr.values())
    assert abs(c_sum - 1) < tol

    orbit_data = cluster_space.orbit_data
    best_score = 1e9

    if include_smaller_cells:
        sizes = list(range(1, max_size + 1))
    else:
        sizes = [max_size]

    prim = cluster_space.primitive_structure
    if pbc is None:
        pbc = (True, True, True)
    prim.set_pbc(pbc)

    for structure in enumerate_structures(prim,
                                          sizes,
                                          cluster_space.chemical_symbols,
                                          concentration_restrictions=cr):
        cv = cluster_space.get_cluster_vector(structure)
        score = compare_cluster_vectors(cv_1=cv,
                                        cv_2=sqs_vector,
                                        orbit_data=orbit_data,
                                        optimality_weight=optimality_weight,
                                        tol=tol)

        if score < best_score:
            best_score = score
            best_structure = structure
    return best_structure


def occupy_structure_randomly(structure: Atoms, cluster_space: ClusterSpace,
                              target_concentrations: dict) -> None:
    #--
    target_concentrations = _validate_concentrations(
        cluster_space=cluster_space, concentrations=target_concentrations)

    if not _concentrations_fit_structure(structure, cluster_space,
                                         target_concentrations):
        raise ValueError('Structure with {} atoms cannot accomodate '
                         'target concentrations {}'.format(
                             len(structure), target_concentrations))

    symbols_all = [''] * len(structure)
    for sl in cluster_space.get_sublattices(structure):
        symbols = []

        chemical_symbols = sl.chemical_symbols
        if len(chemical_symbols) == 1:
            symbols += [chemical_symbols[0]] * len(sl.indices)
        else:
            sl_conc = target_concentrations[sl.symbol]
            for chemical_symbol in sl.chemical_symbols:
                n_symbol = int(
                    round(len(sl.indices) * sl_conc[chemical_symbol]))
                symbols += [chemical_symbol] * n_symbol

        assert len(symbols) == len(sl.indices)

        random.shuffle(symbols)

        for symbol, lattice_site in zip(symbols, sl.indices):
            symbols_all[lattice_site] = symbol

    assert symbols_all.count('') == 0
    structure.set_chemical_symbols(symbols_all)


def _validate_concentrations(concentrations: dict,
                             cluster_space: ClusterSpace,
                             tol: float = 1e-5) -> dict:
    #--
    sls = cluster_space.get_sublattices(cluster_space.primitive_structure)

    if not isinstance(list(concentrations.values())[0], dict):
        concentrations = {'A': concentrations}

    for sl_conc in concentrations.values():
        conc_sum = sum(list(sl_conc.values()))
        if abs(conc_sum - 1.0) > tol:
            raise ValueError(
                'Concentrations must sum up '
                'to 1 for each sublattice (not {})'.format(conc_sum))

    for sl in sls:
        if sl.symbol not in concentrations:
            if len(sl.chemical_symbols) > 1:
                raise ValueError('A sublattice ({}: {}) is missing in '
                                 'target_concentrations'.format(
                                     sl.symbol, list(sl.chemical_symbols)))
        else:
            sl_conc = concentrations[sl.symbol]
            if tuple(sorted(sl.chemical_symbols)) != tuple(
                    sorted(list(sl_conc.keys()))):
                raise ValueError(
                    'Chemical symbols on a sublattice ({}: {}) are '
                    'not the same as those in the specified '
                    'concentrations {}'.format(sl.symbol,
                                               list(sl.chemical_symbols),
                                               list(sl_conc.keys())))

    return concentrations


def _concentrations_fit_structure(structure: Atoms,
                                  cluster_space: ClusterSpace,
                                  concentrations: Dict[str, Dict[str, float]],
                                  tol: float = 1e-5) -> bool:
    #--

    for sublattice in cluster_space.get_sublattices(structure):
        if sublattice.symbol in concentrations:
            sl_conc = concentrations[sublattice.symbol]
            for conc in sl_conc.values():
                n_symbol = conc * len(sublattice.indices)
                if abs(int(round(n_symbol)) - n_symbol) > tol:
                    return False
    return True


def _get_sqs_cluster_vector(
        cluster_space: ClusterSpace,
        target_concentrations: Dict[str, Dict[str, float]]) -> np.ndarray:
    #--
    target_concentrations = _validate_concentrations(
        concentrations=target_concentrations, cluster_space=cluster_space)

    sublattice_to_index = {
        letter: index
        for index, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    }
    all_sublattices = cluster_space.get_sublattices(
        cluster_space.primitive_structure)

    symbol_to_integer_map = {}
    found_species = []

    for sublattice in all_sublattices:
        if len(sublattice.chemical_symbols) < 2:
            continue
        atomic_numbers = [
            periodic_table.index(sym) for sym in sublattice.chemical_symbols
        ]
        for i, species in enumerate(sorted(atomic_numbers)):
            found_species.append(species)
            symbol_to_integer_map[periodic_table[species]] = i

    probabilities = {}
    for sl_conc in target_concentrations.values():
        if len(sl_conc) == 1:
            continue
        for symbol in sl_conc.keys():
            probabilities[symbol] = sl_conc[symbol]

    cv = [1.0]
    for orbit in cluster_space.orbit_data:
        if orbit['order'] < 1:
            continue

        sublattices = [
            all_sublattices[sublattice_to_index[letter]]
            for letter in orbit['sublattices'].split('-')
        ]

        symbol_groups = [
            sublattice.chemical_symbols for sublattice in sublattices
        ]

        nbr_of_allowed_species = [
            len(symbol_group) for symbol_group in symbol_groups
        ]

        cluster_product_average = 0
        for symbols in itertools.product(*symbol_groups):
            cluster_product = 1
            for i, symbol in enumerate(symbols):
                mc_vector_component = orbit['multi_component_vector'][i]
                species_i = symbol_to_integer_map[symbol]
                prod = cluster_space.evaluate_cluster_function(
                    nbr_of_allowed_species[i], mc_vector_component, species_i)
                cluster_product *= probabilities[symbol] * prod
            cluster_product_average += cluster_product
        cv.append(cluster_product_average)
    return np.array(cv)
