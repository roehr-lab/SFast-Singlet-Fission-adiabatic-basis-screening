"""
helper functions for reading and writing xyz-files
"""
import torch

# unit conversions
hartree_to_eV = 27.211396132

def read_xyz(filename):
    """

    reads all geometries from an XYZ-file, the order and type of the atoms
    should be the same in all geometries

    Parameters
    ----------
    filename   :   path to xyz-file

    Returns
    -------
    species    :   Tensor (shape (nmol, molsize))
       atomic numbers in descending order
    coordinates :  Tensor (shape (nmol, molsize, 3))
       cartesian coordinates in Angstrom
    unsort_indices : list of lists
       indices for reverting to the atom order in the xyz-file

    """
    # mapping from element names to atomic numbers
    elem2num = { 'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8  }
    molecules = []
    with open(filename) as f:
        try:
            while True:
                # Each molecule is a list of tuples containing the atom numbers
                # and the cartesian positions in Angstrom, i.e. `(atnum,(x,y,z))`
                molecule = []
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
                    molecule.append( (Zat, (x,y,z)) )
                molecules.append(molecule)
        except EOFError:
            assert len(molecule) == nat
    # number of molecules
    nmol = len(molecules)
    # `molsize` is the size of the biggest molecule.
    # Molecules with less atoms are padded with zeros
    molsize = max([len(molecule) for molecule in molecules])

    # fill in arrays of species and coordinates
    species = torch.zeros((nmol,molsize),dtype=torch.int64)
    coordinates = torch.zeros((nmol,molsize,3))
    unsort_indices = []

    for k in range(0, nmol):
        # PYSEQM expects that the atoms are sorted
        # by atomic number in decreasing order. 
        molecule = molecules[k]
        
        # sort atoms by element in decreasing order
        sort_idx = torch.argsort(torch.tensor([Zat for (Zat,pos) in molecule]), descending=True)
        # reverse mapping from sorted indices to original ordering
        unsort_idx = torch.zeros(molsize, dtype=torch.int64)
        for i in range(0, molsize):
            unsort_idx[sort_idx[i]] = i
        unsort_indices.append(unsort_idx)
            
        # Now the atoms are sorted by atomic number
        molecule = [molecule[i] for i in sort_idx]

        for i in unsort_idx:
            Zat,(x,y,z) = molecule[i]
            species[k,i] = Zat
            coordinates[k,i,:] = torch.tensor([x,y,z])

    return species, coordinates, unsort_indices

def write_xyz(filename, species, coordinates, unsort_indices=None, mode="w", title=""):
    """

    save geometries in XYZ format

    Parameters
    ----------
    filename    :   str
       path to molden output file
    species     :   int Tensor of shape (nmol, molsize)
       atomic numbers
    coordinates :   float Tensor of shape (nmol, molsize, 3)
       cartesian coordinates in Angstrom
    unsort_indices :  int Tensor of shape (nmol, molsize)
       PYSEQM requires atoms to be sorted by atomic numbers in decreasing order.
       `unsort_indices` allows to reverse this sorting, so that the atoms are
       written in the original order. If None, no reordering is performed.
    mode           : str
       "a" - append to existing file
       "w" - overwrite existing file
    title          : str
       title line

    """
    nmol, molsize = species.size()
    # mapping from atomic numbers to element names
    num2elem = { 1 : 'H', 5 : 'B', 6 : 'C', 7 : 'N', 8 : 'O' }
    with open(filename, mode) as f:
        for k in range(0, nmol):
            # Count number of atoms without 0's from padding
            nat = (species[k,:] > 0).sum()
            f.write("%d\n" % nat)
            f.write("%s\n" % title)
            for i in range(0, nat):
                if unsort_indices is None:
                    iu = i
                else:
                    iu = unsort_indices[k][i]
                f.write(" %2s  %+12.8f    %+12.8f    %+12.8f  \n" % (num2elem[species[k,iu].item()],
                                                                      coordinates[k,iu,0].item(),
                                                                      coordinates[k,iu,1].item(),
                                                                      coordinates[k,iu,2].item()))

"""
STO-3G basis set for atoms H,B,C,N,O.
Only basis functions for the valence orbitals are included
"""
STO3G_valence = \
{
# HCNO
 "h": 
" s    3 1.00                                  \n\
 0.34252510000000E+01 0.15432897017315E+00     \n\
 0.62391374000000E+00 0.53532814060062E+00     \n\
 0.16885541000000E+00 0.44463454049886E+00     \n\
",
 "b":
" s    3 1.00                                  \n\
 0.2236956142E+01      -0.9996722919E-01       \n\
 0.5198204999E+00       0.3995128261E+00       \n\
 0.1690617600E+00       0.7001154689E+00       \n\
  p    3 1.00                                  \n\
 0.2236956142E+01       0.1559162750E+00       \n\
 0.5198204999E+00       0.6076837186E+00       \n\
 0.1690617600E+00       0.3919573931E+00       \n\
",
 "c": 
" s    3 1.00                                   \n\
 0.29412494000000E+01 -.99967228413962E-01     \n\
 0.68348310000000E+00 0.39951282765793E+00     \n\
 0.22228992000000E+00 0.70011546589571E+00     \n\
 p    3 1.00                                   \n\
 0.29412494000000E+01 0.15591627960143E+00     \n\
 0.68348310000000E+00 0.60768371844656E+00     \n\
 0.22228992000000E+00 0.39195738899803E+00     \n\
",
 "n": 
" s    3 1.00                                   \n\
 0.37804559000000E+01 -.99967228569568E-01     \n\
 0.87849664000000E+00 0.39951282827980E+00     \n\
 0.28571437000000E+00 0.70011546698549E+00     \n\
 p    3 1.00                                   \n\
 0.37804559000000E+01 0.15591628002847E+00     \n\
 0.87849664000000E+00 0.60768372011097E+00     \n\
 0.28571437000000E+00 0.39195739007157E+00     \n\
",
 "o": 
" s    3 1.00                                   \n\
 0.50331513000000E+01 -.99967228336652E-01     \n\
 0.11695961000000E+01 0.39951282734897E+00     \n\
 0.38038896000000E+00 0.70011546535427E+00     \n\
 p    3 1.00                                   \n\
 0.50331513000000E+01 0.15591627951715E+00     \n\
 0.11695961000000E+01 0.60768371811808E+00     \n\
 0.38038896000000E+00 0.39195738878615E+00     \n\
",
# sulphur Turbomole sto-3g basis, 3s and 3p orbitals
 "s":
" s    3 1.0                                   \n\
 2.0291942740          -0.2196203690           \n\
 0.5661400518           0.2255954336           \n\
 0.2215833792           0.9003984260           \n\
 p     3 1.0                                   \n\
 2.0291942740           0.01058760429          \n\
 0.5661400518           0.59516700530          \n\
 0.2215833792           0.46200101200          \n\
",
# Zn (from bse.pnl.gov/bse/portal  Turbomole STO-3G basis)  4d, 5s and 5p orbitals 3d and 4s orbitals
 "zn":
" d   3 1.0                                    \n\
 10.94737077            0.2197679508           \n\
 3.339297018            0.6555473627           \n\
 1.288404602            0.2865732590           \n\
 s    3 1.0                                    \n\
 0.8897138854          -0.3088441215           \n\
 0.3283603790           0.0196064117           \n\
 0.1450074055           1.1310344420           \n\
",
# silver (from bse.pnl.gov/bse/portal  Turbomole STO-3G basis)  4d, 5s and 5p orbitals
 "ag":
" d   3 1.0                                    \n\
 3.283395668            0.1250662138           \n\
 1.278537254            0.6686785577           \n\
 0.5628152469           0.3052468245           \n\
 s    3 1.0                                    \n\
 0.4370804803          -0.3842642607           \n\
 0.2353408164          -0.1972567438           \n\
 0.1039541771           1.3754955120           \n\
 p    3 1.0                                    \n\
 0.4370804803          -0.3481691526           \n\
 0.2353408164           0.6290323690           \n\
 0.1039541771           0.6662832743           \n\
"
}
                
def write_molden(filename, species, coordinates, orbs, orbe, nocc,
                 mol_index=0,
                 unsort_indices=None, title="PYSEQM"):
    """

    save geometry and MO coefficients in Molden format

    Parameters
    ----------
    filename    :   str
       path to molden output file
    species     :   int Tensor of shape (nmol, molsize)
       atomic numbers
    coordinates :   float Tensor of shape (nmol, molsize, 3)
       cartesian coordinates in Angstrom
    orbs        :   float Tensor of shape (nmol, nao, nmo)
       molecular orbital coefficients
    orbe        :   float Tensor of shape (nmol, nmo)
       orbital energies
    nocc        :   int Tensor of shape (nmol,)
       number of occupied orbitals
    mol_index   :   0 <= int <= nmol
       Index of the molecule for which the coordinates and orbitals 
       should be exported. Only one molecule can be exported per file.
    unsort_indices :  int Tensor of shape (nmol, molsize)
       PYSEQM requires atoms to be sorted by atomic numbers in decreasing order.
       `unsort_indices` allows to reverse this sorting, so that the atoms are
       written in the original order. If None, no reordering is performed.
    title          : str
       title line

    """
    nmol, molsize = species.size()
    nmol, nao, nmo = orbs.size()
    """
    print("number of molecules : %d" % nmol)
    print("number of MOs       : %d" % nmo)
    print("number of AOs       : %d" % nao)
    """
    # mapping from atomic numbers to element names
    num2elem = { 1 : 'H', 5 : 'B', 6 : 'C', 7 : 'N', 8 : 'O' }
    # only first geometry is written
    k = 0
    with open(filename, "w") as f:
        f.write("[Molden Format]\n")
        f.write("[Title]\n")
        f.write("%s\n" % title)
        # ATOMS
        # Count number of atoms without 0's from padding
        nat = (species[k,:] > 0).sum()
        f.write("[Atoms] Ang\n")
        for i in range(0, nat):
            if unsort_indices is None:
                iu = i
            else:
                iu = unsort_indices[k][i]
            atnum = species[k,iu].item()
            f.write(" %2s     %4.d     %2.d       %12.8f %12.8f %12.8f\n" \
                % (num2elem[atnum], i+1, atnum,
                   coordinates[k,iu,0].item(),
                   coordinates[k,iu,1].item(),
                   coordinates[k,iu,2].item()) )
        # GTO
        f.write("[GTO]\n")
        for i in range(0, nat):
            if unsort_indices is None:
                iu = i
            else:
                iu = unsort_indices[k][i]
            elem = num2elem[species[k,iu].item()]
            f.write("%2.d 0\n" % (i+1))
            f.write(STO3G_valence[elem.lower()])
        # MO
        f.write("[5D]\n")
        f.write("[MO]\n")
        for mo in range(0, nmo):
            f.write(" Sym= %sa\n" % (mo+1))
            f.write(" Ene= %12.8f \n" % (orbe[k,mo] / hartree_to_eV))
            f.write(" Spin= Alpha \n")
            # occupation numbers
            if mo < nocc[k]:
                occ = 2.0
            else:
                occ = 0.0
            f.write(" Occup= %8.4f\n" % occ)
            for ao in range(0, nao):
                f.write("   %4.d    %12.8f \n" % (ao+1, orbs[k,ao,mo]))
        
