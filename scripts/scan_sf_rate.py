#!/usr/bin/env python
"""
The singlet fission rate is computed using the frontier orbital model [1]
and scanned as a function of the relative displacement of two planar
molecules along the x and y axes (Each molecule lies in the xy plane, the
distance along the z-axis is 3.5 Ang).

The scan should reproduce Fig. 4 of [1] for ethylene and Fig. 3 of [2] for PDI.

requires PYSEQM (https://github.com/lanl/PYSEQM)

References
----------
[1] Buchanan et. al. (2017)
    "Singlet Fission: Optimization of Chromophore Dimer Geometry"
    http://dx.doi.org/10.1016/bs.aiq.2017.03.005
[2] K. Felter, F. Grozema,
    "Singlet Fission in Crystalline Organic Materials: Recent Insights and Future Directions"
    http://dx.doi.org/10.1021/acs.jpclett.9b00754

"""
import torch
import tqdm
import argparse

from PYSEQM import seqm
from PYSEQM.seqm.basics import Parser, Pack_Parameters, Hamiltonian
from PYSEQM.seqm.seqm_functions.diat_overlap import diatom_overlap_matrix
from PYSEQM.seqm.seqm_functions import constants
from PYSEQM.seqm.seqm_functions.diag import sym_eig_trunc1

from singlet_fission import pyseqm_helpers
from singlet_fission.SingletFissionRate import SingletFissionRate

# CPU or GPU?
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
        
def read_xyz(filename):
    """
    reads all geometries from an XYZ-file, the order and type of the atoms
    should be the same in all geometries

    Parameters
    ----------
    filename   :   path to xyz-file

    Returns
    -------
    atomlists  :   list of geometries
      Each geometry is a list of tuples containing the atom numbers and the cartesian positions
      in bohr, i.e. `(atnum,(x,y,z))`

    """
    # mapping from element names to atomic numbers
    elem2num = { 'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8  }
    atomlists = []
    with open(filename) as f:
        try:
            while True:
                atomlist = []
                # number of atoms
                line = f.readline()
                if line == "":
                    break
                nat = int(line)
                # title line
                title = f.readline()
                for i in range(0, nat):
                    parts = f.readline().strip().split()
                    Zat = elem2num[parts[0].upper()]
                    x,y,z = map(lambda v: float(v), parts[1:4])
                    atomlist.append( (Zat, (x,y,z)) )
                atomlists.append(atomlist)
        except EOFError:
            assert len(atomlist) == nat
    return atomlists

import sys
import os.path

parser = argparse.ArgumentParser(
    description="""
The matrix element for singlet fission |T_RP|^2 is computed using different 
semiempirical approximations and scanned as a function of the relative displacement 
of two planar molecules along the x and y axes (Each molecule lies in the xy plane, 
the distance along the z-axis is 3.5 Ang). The absolute values of |T_RP|^2 are 
meaningless, only the positions of the maxima are important.
""")

parser.add_argument('xyz_file',
                    type=str, metavar='monomer.xyz',
                    help='XYZ file with geometry of monomer (in xy-plane) used to build the homo dimer')
parser.add_argument('--approximation',
                    type=str, metavar='model',
                    help='Choose approximation for calculating |T_RP|^2. "diabatic" : SF rate is approximated by the diabatic coupling between the singlet exciton S and the biexciton TT states, which can both mix with higher-lying charge transfer states. Matrix elements of the Fock operator between monomer frontier orbitals are used. "non-adiabatic": SF rate is approximated as the length squared of the non-adiabatic coupling vector between adiabatic states which have mostly exciton (S*) or biexciton (TT*) character, respectively, |<S*|grad|TT*>|^2',
                    default='diabatic')

args = parser.parse_args()

name=os.path.basename(args.xyz_file).replace(".xyz","")

# Geometry of monomer, xy-plane should be the molecular plane)
atomlists = read_xyz(args.xyz_file)

monomer = atomlists[0]
# We try to reproduce the scan of |T_RP|^2 in Fig. 3 of Felter & Grozema.
# The dimer contains to parallel molecules displaced along the z-axis by 3.5 Ang
dimer = monomer + monomer[:]

# number of atoms 
molsize = len(dimer)

# PYSEQM expects that the atoms are sorted by atomic number in decreasing
# order. However we also need to keep track of which atom index belongs
# to which fragment, so that we can do separate computations for fragment A,
# fragment B and the whole system A+B.

# sort atoms by element in decreasing order
sort_idx = torch.argsort(torch.tensor([Zat for (Zat,pos) in dimer]), descending=True)
# reverse mapping from sorted indices to original ordering
unsort_idx = torch.zeros(molsize, dtype=torch.int64)
for i in range(0, molsize):
    unsort_idx[sort_idx[i]] = i

# Now the atoms are sorted by atomic number
dimer = [dimer[i] for i in sort_idx] 

# atom_indices_A and atom_indices_B contain the indices of the atoms belonging
# to fragments A and B.
# It is assumed that both fragments have the same size (molsize//2) (e.g. ethene or tetracene dimer)
# and that in the original order the atoms of the first fragment come first.

assert molsize % 2 == 0

atom_indices_A = []
atom_indices_B = []
for i in range(0, molsize):
    if i in unsort_idx[:molsize//2]:
        atom_indices_A.append(i)
    elif i in unsort_idx[molsize//2:]:
        atom_indices_B.append(i)

atom_indices_A = torch.tensor(atom_indices_A)
atom_indices_B = torch.tensor(atom_indices_B)

print("indices of atoms of fragment A: %s" % atom_indices_A)
print("indices of atoms of fragment B: %s" % atom_indices_B)
print("atomic numbers: %s" % [atom[0] for atom in dimer])

# create the grid of relative displacements between the dimers
# for scanning |T_RP|^2(x,y,z)
#  x in [-4,4] Ang
#  y in [-4,4] Ang
#  z = 3.5 Ang

# PYSEQM uses units of Angstrom for coordinates
dxx,dyy = torch.meshgrid(torch.arange(-4.0,4.0,0.1), torch.arange(-4.0,4.0,0.1))
dz = 3.5

# turn 2D arrays into 1D arrays
dx,dy = dxx.reshape(-1), dyy.reshape(-1)

# number of geometries in the scan
nmol = dx.shape[0]

species = torch.ones((nmol,molsize),dtype=torch.int64, device=device)
coordinates = torch.zeros((nmol,molsize,3), device=device)

# generate geometries for each scan point
for k in range(0, nmol):
    # First molecule is always placed at the same position x,y,z
    for i in atom_indices_A:
        Zat,(x,y,z) = dimer[i]
        # Reflect the first monomer at the xy-plane
        z = -z
        species[k,i] = Zat
        coordinates[k,i,:] = torch.tensor([x,y,z])
    # The second molecule is shifted to position x+dx,y+dy,z+dz
    for i in atom_indices_B:
        Zat,(x,y,z) = dimer[i]
        species[k,i] = Zat
        coordinates[k,i,:] = torch.tensor([x+dx[k],y+dy[k],z+dz])

print("atomic numbers of fragment A: ", [species[0,i].item() for i in atom_indices_A])
print("atomic numbers of fragment B: ", [species[0,i].item() for i in atom_indices_B])

# save all geometries in the scan
pyseqm_helpers.write_xyz("scan.xyz", species, coordinates)

elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
    'method' : 'AM1',  # AM1, MNDO, PM#
    'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
    'scf_converger' : [2,0.0], # converger used for scf loop
    # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
    # [1], adaptive mixing
    # [2], adaptive mixing, then pulay
    'sp2' : [True, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
    #[True, eps] or [False], eps for SP2 conve criteria
    'elements' : elements, #[0,1,6,8],
    'learned' : [],
    # file directory for other required parameters
    'parameter_file_dir' :  os.path.normpath(os.path.join(os.path.dirname(seqm.__file__),"../params/MOPAC"))+"/",
    'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
}

rates = torch.zeros(nmol)

# list of geometries is split into nc chunks that are processed in parallel
nc = nmol//128

# Which of the different approximations for the SF rate in Ref.[1] should be used?
approximation = args.approximation #'non-adiabatic' #'diabatic' #'overlap'

print(f"Calculation will be run on device '{device}'")
print(f"approximation for |T_RP|^2 : {approximation}")

with torch.autograd.set_detect_anomaly(True):
    with tqdm.tqdm(total=nc) as progress_bar:
        for ic, (species_, coordinates_, rates_) in enumerate(
                zip(torch.chunk(species,nc,dim=0),
                    torch.chunk(coordinates,nc,dim=0),
                    torch.chunk(rates,nc,dim=0))):

            if approximation == 'non-adiabatic':
                coordinates_.requires_grad_(True)
            sfr = SingletFissionRate(seqm_parameters, species_,
                                     atom_indices_A, atom_indices_B,
                                     approximation=approximation).to(device)
            # compute rates
            rates_[:] = sfr(coordinates_).cpu()

            # show progress
            progress_bar.set_description(f"forward pass for {len(species_)} geometries (chunk {ic+1} out of {nc})")
            progress_bar.update(1)

# Find geometry with largest SF rate
imax = torch.argmax(rates).item()
sfr = SingletFissionRate(seqm_parameters, species[imax,:].unsqueeze(0),
                         atom_indices_A, atom_indices_B,
                         approximation=approximation).to(device)

# orbitals of dimer and hA,lA, hB,lB at the geometry with largest SF coupling
sfr.save_dimer_orbitals("dimer_orbitals_max_rate.molden", coordinates[imax,:,:].unsqueeze(0))
sfr.save_monomer_orbitals("monomer_orbitals_max_rate.molden", coordinates[imax,:,:].unsqueeze(0))

# save scan as .npz file.
import numpy as np
np.savez("%s_SF-%s.npz" % (name, approximation), dx=dx, dy=dy, dz=dz, approximation=approximation, rates=rates.detach().numpy())

