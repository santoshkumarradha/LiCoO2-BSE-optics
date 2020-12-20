from ase import Atoms
from gpaw import GPAW, PW, FermiDirac, restart
from ase.io import read
from ase.parallel import parprint
import os.path as check
import pickle

if not check.isfile('coo_gs.gpw'):
    atom = read("sqs_coo.cif")
    mmom = []
    atoms_str = []
    for i in atom.get_chemical_symbols():
        if i == "Co":
            val = 1
        if i == "Mn":
            val = -1
            i = "Co"
        if i == "O":
            val = 0
        if i == "Li":
            val = 0
        mmom.append(val)
        atoms_str.append(i)
    atom.set_chemical_symbols(atoms_str)
    atom.set_initial_magnetic_moments(mmom)

    k = 3  # number of k-points
    calc = GPAW(
        mode='lcao',
        basis='dzp',
        occupations=FermiDirac(width=0.05),
        setups={'Co': ':d,5.0'},  # U=5 eV for Co d orbitals
        txt='Coo.txt',
        kpts=(k, k, k),
        nbands='nao',
        xc='PBE')
    occasionally = 10

    class OccasionalWriter:
        def __init__(self):
            self.iter = 0

        def write(self):
            calc.write('coo_gs.%03d.gpw' % self.iter)
            self.iter += occasionally

    calc.attach(OccasionalWriter().write, occasionally)
    atom.set_calculator(calc)
    e = atom.get_potential_energy()
    calc.write('coo_gs.gpw')
    parprint("total energy = {}".format(e))

    calc.set(nbands='nao',
             fixdensity=True,
             symmetry='off',
             kpts={
                 'path': atom.cell.bandpath().path,
                 'npoints': 200
             },
             convergence={'bands': 'all'})

    calc.get_potential_energy()
    calc.write('coo_bands.gpw')
    bs = calc.band_structure()
    with open('band_energies.pickle', 'wb') as handle:
        pickle.dump(calc.band_structure().todict(),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    bs.plot(filename='bandstructure.png',
            show=False,
            emax=8.0,
            emin=-8.0,
            reference=8.84354)

else:
    atom = read("sqs_coo.cif")
    parprint("Reading from restart file")
    calc = GPAW('coo_gs.gpw',
                nbands='nao',
                fixdensity=True,
                symmetry='off',
                kpts={
                    'path': atom.cell.bandpath().path,
                    'npoints': 200
                },
                convergence={'bands': 'CBM+6'})
    calc.get_potential_energy()
    calc.write('coo_bands.gpw')
    bs = calc.band_structure()
    bs.plot(filename='bandstructure.png',
            show=False,
            emax=8.0,
            emin=-8.0,
            reference=8.84354)
