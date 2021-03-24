"""
singlet fission rate according to Michl's model
"""
import torch
import numpy as np
from torch import optim
from torch import autograd

from PYSEQM import seqm
from PYSEQM.seqm.basics import Parser, Pack_Parameters, Hamiltonian, Energy
from PYSEQM.seqm.seqm_functions.diat_overlap import diatom_overlap_matrix
from PYSEQM.seqm.seqm_functions import constants
from PYSEQM.seqm.seqm_functions.diag import sym_eig_trunc1
from PYSEQM.seqm.seqm_functions.pack import pack

from singlet_fission import pyseqm_helpers

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    print("CUDA available")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def softmax(y, dim=0, beta=30.0):
    """
    differentiable substitute for argmax(y)
    
    max(y) = softmax(y)*y
    
    Example
    -------
    >>> y = torch.tensor([0.7, 0.99, 0.9, 0.1])
    >>> softmax(y)
    tensor([1.5607e-04, 9.3688e-01, 6.2963e-02, 2.3770e-12])
    >>> y * softmax(y)
    tensor([1.0925e-04, 9.2751e-01, 5.6667e-02, 2.3770e-13])
    """
    a = torch.exp(beta*y)
    b = torch.sum(a, dim).unsqueeze(dim)
    return a/b


class SingletFissionRate(torch.nn.Module):
    def __init__(self, seqm_parameters, species, atom_indices_A, atom_indices_B,
                 approximation='overlap',
                 exciton_state='both'):
        """
        computes |T_RP|^2 for a dimer

        Parameters
        ----------
        atom_indices_A :  Tensor (int,)
          indices of atoms belonging to monomer A
        atom_indices_B :  Tensor (int,)
          indices of atoms belonging to monomer B
        approximation  :  str
          Choose approximation for |T_RP|^2, 'diabatic' or 'non-adiabatic'
        exciton_state  :  str
          Initial excitonic state for singlet fission, either the bright state ('bright')
          or the dark state ('dark'). In case of 'both', the matrix element for singlet
          fission |T_RP|^2 is the incoherent average for the bright and dark states.
        """
        super().__init__()
        
        self.const = constants.Constants().to(device)
        self.species = species.to(device)
        self.atom_indices_A = atom_indices_A.to(device)
        self.atom_indices_B = atom_indices_B.to(device)
        self.parser = Parser(seqm_parameters)
        self.packpar = Pack_Parameters(seqm_parameters)
        self.hamiltonian = Hamiltonian(seqm_parameters)

        assert approximation in ['overlap', 'diabatic', 'non-adiabatic']
        self.approximation = approximation
        assert exciton_state in ['bright', 'dark', 'both']
        self.exciton_state = exciton_state

        # number of valence orbital for elements H,C,N and O
        self.num_valorbs = {1 : 1,
                            5 : 4,
                            6 : 4,
                            7 : 4,
                            8 : 4}
        
    def forward(self, coordinates):
        """
        simple approximations for the singlet fission rate according to [1]

        References
        ----------
        [1] A. Buchanan et.al.
            "Singlet Fission: Optimization of Chromophore Dimer Geometry"
            http://dx.doi.org/10.1016/bs.aiq.2017.03.005
        """
        # MO coefficients of frontier orbitals
        hA,lA = self.frontier_orbitals_subsystem(coordinates, self.atom_indices_A)
        hB,lB = self.frontier_orbitals_subsystem(coordinates, self.atom_indices_B)

        if self.approximation == "overlap":
            # approximation according to eqn. (43) in [1]
            
            # AO overlap matrix between orbitals on fragments A and B, S_AB
            S_AB = self.overlap_AB(coordinates)
                
            nmol, naoA, naoB = S_AB.size()
        
            S = torch.zeros((nmol,2,2)).to(device)
            # Shh = S[0,0], Shl = S[0,1], Slh = S[1,0], Sll = S[1,1]

            for m in range(0, naoA):
                for n in range(0, naoB):
                    S[...,0,0] += hA[...,m]*hB[...,n]*S_AB[...,m,n]
                    S[...,0,1] += hA[...,m]*lB[...,n]*S_AB[...,m,n]
                    S[...,1,0] += lA[...,m]*hB[...,n]*S_AB[...,m,n]
                    S[...,1,1] += lA[...,m]*lB[...,n]*S_AB[...,m,n]
                    
            t2 = (S[...,1,0]*S[...,1,1]-S[...,0,1]*S[...,0,0])**2

            if nmol == 1:
                print("Shh= %e  Shl= %e  Slh= %e  Sll= %e" % (S[0,0,0].item(), S[0,0,1].item(), S[0,1,0].item(), S[0,1,1].item()))
            
        elif self.approximation == "diabatic":
            # approximation according to eqn. (39) in [2]

            # Fock matrix between orbitals fragments A and B, F_AB
            F_AB = self.fock_matrix_AB(coordinates)
            nmol, naoA, naoB = F_AB.shape
        
            F = torch.zeros((nmol,2,2)).to(device)
            # Fhh = F[0,0], Fhl = F[0,1], Flh = F[1,0], Fll = F[1,1]

            for m in range(0, naoA):
                for n in range(0, naoB):
                    F[...,0,0] += hA[...,m]*hB[...,n]*F_AB[...,m,n]
                    F[...,0,1] += hA[...,m]*lB[...,n]*F_AB[...,m,n]
                    F[...,1,0] += lA[...,m]*hB[...,n]*F_AB[...,m,n]
                    F[...,1,1] += lA[...,m]*lB[...,n]*F_AB[...,m,n]

            t2 = (F[...,1,0]*F[...,1,1]-F[...,0,1]*F[...,0,0])**2

            if nmol == 1:
                print("Fhh= %e  Fhl= %e  Flh= %e  Fll= %e" % (F[0,0,0].item(), F[0,0,1].item(), F[0,1,0].item(), F[0,1,1].item()))

        elif self.approximation == "non-adiabatic":
            # Fock matrix between orbitals fragments A and B, F_AB
            F_AB = self.fock_matrix_AB(coordinates)
            nmol, naoA, naoB = F_AB.shape
        
            F = torch.zeros((nmol,2,2)).to(device)
            # Fhh = F[0,0], Fhl = F[0,1], Flh = F[1,0], Fll = F[1,1]

            for m in range(0, naoA):
                for n in range(0, naoB):
                    F[...,0,0] += hA[...,m]*hB[...,n]*F_AB[...,m,n]
                    F[...,0,1] += hA[...,m]*lB[...,n]*F_AB[...,m,n]
                    F[...,1,0] += lA[...,m]*hB[...,n]*F_AB[...,m,n]
                    F[...,1,1] += lA[...,m]*lB[...,n]*F_AB[...,m,n]

            # Simply rename matrix elements of Fock matrix
            # <hA|F|hB>
            tHH = F[...,0,0]
            # <hA|F|lB>
            tHL = F[...,0,1]
            # <lA|F|hB>
            tLH = F[...,1,0]
            # <lA|F|lB>
            tLL = F[...,1,1]

            #
            # Here we construct the diabatic Hamiltonian in the basis of the direct product
            # of monomer states, |S0S1>, |S1S0>, |AC>, |CA>, |T1T1>
            # see eqn. (9) in
            #
            #   Mirjani, F. et.al. 
            #   "Theoretical Investigation of Singlet Fission in Molecular Dimers: The
            #    Role of Charge Transfer States and Quantum Interference"
            #   dx.doi.org/10.1021/jp503398a | J. Phys. Chem. C 2014, 118, 14192−14199
            #
            H = torch.zeros((nmol,5,5)).to(device)
            # E(S0S1) = E(S1S0) = E(T1T1) = 0.0
            # E(AC)= E(CA) = 1.0 eV
            H[...,0,0] = 0.0
            H[...,1,1] = 0.0
            H[...,2,2] = 1.0
            H[...,3,3] = 1.0
            H[...,4,4] = 0.1

            # H-like excitonic coupling, J_ge > 0
            #  Jge = 2*(hA,lA|lB,hB)
            # unfortunately so far we don't have the two-electron integrals, so
            # we have to guess the coupling
            Jge = 0.0
            H[...,0,1] = Jge
            H[...,1,0] = Jge
            
            # off-diagonal matrix elements, t2e ~ 0
            H[...,0,2] =  tLL
            H[...,0,3] = -tHH
            
            H[...,1,2] = -tHH
            H[...,1,3] =  tLL

            H[...,2,0] =  tLL
            H[...,2,1] = -tHH
            H[...,2,4] = np.sqrt(3.0/2.0) * tHL
            
            H[...,3,0] = -tHH
            H[...,3,1] =  tLL
            H[...,3,4] = np.sqrt(3.0/2.0) * tLH
            
            H[...,4,2] = np.sqrt(3.0/2.0) * tHL
            H[...,4,3] = np.sqrt(3.0/2.0) * tLH

            """
            ### DEBUG
            print("diabatic Hamiltonian")
            print(H)
            ###
            """

            # diagonalize symmetric diabatic matrix
            # Back propagation is only stable if all eigenvalues are different.
            evals, evecs = torch.symeig(H, eigenvectors=True)

            # Singlet fission is a non-adiabatic transition between an exciton state
            # and the paired triplet state. There should be two exciton states,
            # the bright state, 
            #  |S*> =  (|S0S1> + |S1S0>)/sqrt(2)
            # for which the transition dipole moments of both monomers
            # are parallel,
            #          ---->
            #          ---->
            # and the dark exciton state, 
            #  |S**> = (|S0S1> - |S1S0>)/sqrt(2)
            # where the transition dipole moments cancel,
            #          ---->
            #          <----
            # Because the monomers are parallel (H-type coupling) the bright
            # exciton state is higher in energy than the dark one. The absorption
            # of a photon should initially excite |S*> followed by internal conversion
            # to |S**>. |S**> can then undergo singlet fission to the biexciton state
            #  |TT> = |T1T1>
            # The adiabatic states also contain admixtures of charge transfer states.
            # A general adiabatic state |X> is a linear combination of all diabatic states:
            #  |X> = C     |S0S1>  +  C    |S1S0>  +  ... + C     |T1T1>
            #         S0S1             S1S0                  T1T1
            # To identify the adiabatic states with mostly |S*> (or |S**>) character
            # we compute the projections
            #  <S*|X> = (C     +  C     ) / sqrt(2)
            #             S0S1     S1S0
            # or
            #  <S**|X> = (C     - C     ) / sqrt(2)
            #              S0S1    S1S0

            # |<S*|X>|^2
            w_bright_exciton = 0.5 * abs(evecs[...,0,:] + evecs[...,1,:])**2
            # |<S**|X>|^2
            w_dark_exciton   = 0.5 * abs(evecs[...,0,:] - evecs[...,1,:])**2
            # |<TT|X>|^2
            w_triplet = abs(evecs[...,4,:])**2
            
            # Since argmax is not differentiable, we use softmax to select only the coefficients
            # where w_# is maximal (coefficients are multiplied by ~1) and to suppress the other
            # coefficients (which are multiplied by ~0). After summing only the maximum value
            # contributes.
            c_bright_exciton = torch.sum(softmax(w_bright_exciton, dim=1).unsqueeze(1) * evecs, 2)
            c_dark_exciton = torch.sum(softmax(w_dark_exciton, dim=1).unsqueeze(1) * evecs, 2)
            c_triplet = torch.sum(softmax(w_triplet, dim=1).unsqueeze(1) * evecs, 2)
            
            # We assume that singlet fission occurs from both the lower dark
            # and the upper bright excition states.
            # Since the transition dipole moment <S0S0|r|S**> = 0, fluorescence
            # does not compete with singlet fission for emission from |S**>.

            # To compute the non-adiabatic coupling vector
            #  NAC = <exciton|d/dx|paired triplets>
            # we have to get the gradients of coefficients
            #      = sum  C(S^*) * d/dx C(TT)
            #           i  i             i
            #
            # k,k' - index of geometry in batch 
            # i    - index of diabatic state i=0,...,4
            # l    - index of cartesian coordinate x (0), y (1) or z (2)
            #                        S*       TT
            #  NAC       =  sum     C      d(C   ) / d(x      )
            #     k',l         k,i   k,i      k,i       k',l
            #
            #                        S*       TT
            #            =  sum     C      d(C   ) / d(x    ) delta     
            #                  k,i   k,i      k,i       k,l        k,k' 
            #
            # This expression has the form of a Jacobian-vector product
            #
            #  NAC       =  sum     v     J
            #     k',l         k,i   k,i   (k,i),(k',l)
            #
            nac_dark, = autograd.grad(
                # output of the differentiated function
                outputs=[c_triplet],
                # input w.r.t which the gradients will be computed
                inputs=[coordinates],
                # The vector 'v' in the Jacobian-vector product
                grad_outputs=[c_dark_exciton],
                retain_graph=True)
            
            # Length of non-adiabatic coupling vector squared
            t2_dark = torch.sum(abs(nac_dark)**2, (1,2))
            #print("|<S*|d/dx|TT>|^2")
            #print(t2_dark)

            nac_bright, = autograd.grad(
                # output of the differentiated function
                outputs=[c_triplet],
                # input w.r.t which the gradients will be computed
                inputs=[coordinates],
                # The vector 'v' in the Jacobian-vector product
                grad_outputs=[c_bright_exciton])

            # Length of NAC vector squared
            t2_bright = torch.sum(abs(nac_bright)**2, (1,2))
            #print("|<TT|d/dx|S*>|^2")
            #print(t2_bright)
            if self.exciton_state == 'bright':
                # Singlet fission happens from the bright exciton state
                # (which is the lower/higher exciton state for J-coupling/H-coupling)
                t2 = t2_bright
            elif self.exciton_state == 'dark':
                # Singlet fission happens from the dark exciton state
                # (which is the higher/lower exciton state for J-coupling/H-coupling)
                t2 = t2_dark
            else:
                # Incoherent average of rates for the channels
                #   |S*>  --> |T1T1>
                #   |S**> --> |T1T1>
                t2 = 0.5*(t2_dark + t2_bright)

        return t2
        
    def overlap_AB(self, coordinates):
        """
        The AO overlap matrix for the combined system A+B has the following structure
        (after reordering atoms)
        
               S_AA  S_AB
           S =
               S_BA  S_BB
        
        We need to extract the block S_AB 
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(self.const, self.species, coordinates)
        parameters = self.packpar(Z)
        zetas = parameters['zeta_s']
        zetap = parameters['zeta_p']
        zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)),dim=1)
        di = diatom_overlap_matrix(ni, nj, xij, rij, zeta[idxi], zeta[idxj],
                                   self.const.qn_int)
        
        # number of unique atom pairs
        npair = (molsize*(molsize-1))//2
        
        # di contains the overlap matrix elements between unique atom pairs (i < j)
        di = di.reshape((nmol,npair,4,4))
    
        # expand the upper triangle in `di` to a full 2d overlap matrix `S`
        S = torch.zeros((nmol,molsize,molsize,4,4)).to(device)
        
        # atom indices of rows and columns 
        row,col = torch.triu_indices(molsize,molsize, offset=1)
        diag = torch.arange(molsize)
        # orbital indices for 4x4 subblocks
        col_orb  = torch.arange(4).unsqueeze(0).expand(4,4).reshape(-1)
        row_orb  = torch.arange(4).unsqueeze(1).expand(4,4).reshape(-1)
        # upper triangle of overlap matrix
        S[:,row,col  ,:, :] = di
        # diagonal of S, AOs are orthonormal
        S[:,diag,diag, 0,0] = 1.0
        S[:,diag,diag, 1,1] = 1.0
        S[:,diag,diag, 2,2] = 1.0
        S[:,diag,diag, 3,3] = 1.0
        # fill in lower triangle, S is symmetric
        for a in row_orb:
            for b in col_orb:
                S[:,col,row, b,a] = S[:,row,col, a,b]

        # lists of atom indices belonging to fragments A and B
        idxA, idxB = self.atom_indices_A, self.atom_indices_B
        
        # count the number of valence orbitals on fragments A and B
        naoA = sum([self.num_valorbs[int(element)] for element in self.species[0,idxA]])
        naoB = sum([self.num_valorbs[int(element)] for element in self.species[0,idxB]])
        # S_AB contains the overlap matrix elements between orbitals on fragment A
        # and fragment B.
        S_AB = torch.zeros((nmol,naoA,naoB)).to(device)

        # `m` enumerates orbitals on fragment A
        m = 0
        for i in range(0, molsize):
            # `n` enumerates orbitals on fragment B
            n = 0
            # number of valence orbitals on atom i (1 AO for H or 4 AOs for C,N,O)
            dm = self.num_valorbs[int(self.species[0,i])]
            for j in range(0, molsize):
                # number of valence orbitals on atom j
                dn = self.num_valorbs[int(self.species[0,j])]
                
                if (i in idxA) and (j in idxB):
                    S_AB[:,m:m+dm,n:n+dn] = S[:,i,j,0:dm,0:dn]
                
                if (j in idxB):
                    n += dn
                
            assert n == naoB
            if (i in idxA):
                m += dm
        assert m == naoA
    
        return S_AB

    def save_dimer_orbitals(self, molden_file, coordinates):
        """save molecular orbitals of the dimer system"""
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(self.const, self.species, coordinates)
        parameters = self.packpar(Z)
        F, e, P, Hcore, w, charge, notconverged = self.hamiltonian(self.const, molsize, nHeavy, nHydro, nocc, Z, maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, parameters)
        # Fock matrix has shape (nmol, molsize*4, molsize*4)
        assert nmol == 1
        
        # diagonalize Fock matrix
        e, v = sym_eig_trunc1(F,nHeavy, nHydro, nocc, eig_only=True)
        # save molecular orbitals
        orbs = torch.stack(v)
        orbe = e
        pyseqm_helpers.write_molden(molden_file, self.species, coordinates, orbs, orbe, nocc)

    def save_monomer_orbitals(self, molden_file, coordinates):
        """save monomer orbitals hA,lA, hB,lB together with the dimer geometry to a molden file"""
        # MO coefficients of frontier orbitals
        hA,lA = self.frontier_orbitals_subsystem(coordinates, self.atom_indices_A)
        hB,lB = self.frontier_orbitals_subsystem(coordinates, self.atom_indices_B)

        # number of atomic orbitals in A and B fragments
        nmol, naoA = hA.size()
        nmol, naoB = hB.size()
        
        # The coefficients of the frontier orbitals for the monomers A and B are combined into
        # a MO matrix for the combined system:
        # 
        #     MOs
        # ( hA 0  lA  0  ) AOs
        # ( 0  hB  0 lB  )
        #
        orbs = torch.zeros((nmol,naoA+naoB,4))
        orbs[:,:naoA,0] = hA
        orbs[:,naoA:,1] = hB
        orbs[:,:naoA,2] = lA
        orbs[:,naoA:,3] = lB

        orbe = torch.zeros((nmol,4))
        nocc = torch.tensor([2]*nmol)

        # In the monom
        atom_indices = torch.cat((self.atom_indices_A, self.atom_indices_B))
        
        pyseqm_helpers.write_molden(molden_file, self.species[:,atom_indices], coordinates[:,atom_indices,:],
                                    orbs, orbe, nocc)
        
    def fock_matrix_AB(self, coordinates):
        """
        The Fock matrix in the AO basis for the combined system A+B has the following structure
        (after reordering atoms)
        
               F_AA  F_AB
           F =
               F_BA  F_BB
        
        We need to extract the block F_AB 
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(self.const, self.species, coordinates)
        parameters = self.packpar(Z)
        F, e, P, Hcore, w, charge, notconverged = self.hamiltonian(self.const, molsize, nHeavy, nHydro, nocc, Z, maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, parameters)
        # Fock matrix has shape (nmol, molsize*4, molsize*4)
        # lists of atom indices belonging to fragments A and B
        idxA, idxB = self.atom_indices_A, self.atom_indices_B

        """
        ### DEBUG
        e, v = sym_eig_trunc1(F,nHeavy, nHydro, nocc, eig_only=True)
        # save molecular orbitals
        orbs = torch.stack(v)
        orbe = e
        pyseqm_helpers.write_molden("test.molden", self.species, coordinates, orbs, orbe, nocc)
        ###
        """

        # PYSEQM uses 4 valence orbitals for any atom, even for hydrogen,
        # although hydrogen has only 1 s-orbital.
        # pack(...) removes the padding for hydrogen (the 3 p-orbitals).
        F_pack = pack(F,nHeavy, nHydro)
        # count the number of valence orbitals on fragments A and B
        naoA = sum([self.num_valorbs[int(element)] for element in self.species[0,idxA]])
        naoB = sum([self.num_valorbs[int(element)] for element in self.species[0,idxB]])
        # F_AB contains the Fock matrix elements between orbitals on fragment A
        # and fragment B.
        F_AB = torch.zeros((nmol,naoA,naoB)).to(device)

        # `m` enumerates orbitals of the whole system
        m = 0
        # `mA` enumerates orbitals on fragment A
        mA = 0
        for i in range(0, molsize):
            # `n` enumerates orbitals of the whole system
            n = 0
            # `nB` enumerates orbitals on fragment B
            nB = 0
            # number of valence orbitals on atom i (1 AO for H or 4 AOs for C,N,O)
            dm = self.num_valorbs[int(self.species[0,i])]
            for j in range(0, molsize):
                # number of valence orbitals on atom j
                dn = self.num_valorbs[int(self.species[0,j])]
                
                if (i in idxA) and (j in idxB):
                    F_AB[:,mA:mA+dm,nB:nB+dn] = F_pack[:,m:m+dm,n:n+dn]

                n += dn
                if (j in idxB):
                    nB += dn
                    
            assert nB == naoB
            
            m += dm
            if (i in idxA):
                mA += dm
            
        assert mA == naoA
    
        return F_AB
    
    def frontier_orbitals_subsystem(self, coordinates, atom_indices):
        """
        compute frontier orbitals for a subsystem
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(self.const, self.species[:,atom_indices], coordinates[:,atom_indices,:])
        parameters = self.packpar(Z)
        F, e, P, Hcore, w, charge, notconverged = self.hamiltonian(self.const, molsize, nHeavy, nHydro, nocc, Z, maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, parameters)
        # e : orbital energies
        # v : MO coefficients (for all molecules and orbitals)
        e, v = sym_eig_trunc1(F,nHeavy, nHydro, nocc, eig_only=True)

        """
        ### DEBUG
        # save molecular orbitals
        orbs = torch.stack(v)
        orbe = e
        pyseqm_helpers.write_molden("test.molden", self.species[:,atom_indices], coordinates[:,atom_indices,:], orbs, orbe, nocc)
        ###
        """
        
        nao,nmo = v[0].shape
        
        # MO coefficients of HOMO and LUMO
        homo = torch.zeros((nmol, nao)).to(device)
        lumo = torch.zeros((nmol, nao)).to(device)
        for i in range(0, nmol):
            # If nocc is the number of occupied orbitals, nocc-1 should be the index of
            # the HOMO and nocc the index of the LUMO (using 0-based indices).
            nHOMO = nocc[i]-1
            nLUMO = nocc[i]
            homo[i,:] = v[i][:,nHOMO]
            lumo[i,:] = v[i][:,nLUMO]
            
        return homo,lumo
