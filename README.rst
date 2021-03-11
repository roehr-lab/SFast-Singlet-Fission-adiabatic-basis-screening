singlet_fission
---------------

This python module implements two semiempirical models for estimating singlet fission rates
in cofacially stacked homodimers, so that the optimal packing arrangement can be identified.

-----------
Description
-----------
The program constructs the cofacially stacked homodimer by duplicating the monomer
and displacing the copy by the vector (dX,dY,dZ). The vertical distance is kept constant
at dZ=3.5 Ang while the parallel displacement dX and dY are scanned on a rectangular grid
[-4,+4] x [-4,+4] Ang. For each displacement the matrix element |T_RP|^2, which is proportional
to the singlet fission rate is computed using one of the following approximations:

 * *diabatic* model, |T_RP|^2 = |<S0S1|H|TT>|^2, where S0S1 and TT are quasi-diabatic
   exciton and biexciton states (see [Simple]_)
 * *non-adiabatic* model |T_RP|^2 = |<S*|grad|TT*>|^2, where S* and TT* are adiabatic states
   which have mostly exciton and biexciton character, respectively (see [Doping]_). 

The wavefunctions contain 4 electrons in the 4 frontier orbitals, which are the HOMOs and LUMOs on each monomer. 
The molecular orbitals of the monomer and their interactions are computed using the semiempirical AM1 method
as implemented in the [PYSEQM]_ package. The derivatives needed to evaluated the non-adiabatic coupling vector
are computed by automatic differentiation.

------------
Requirements
------------

Required python packages:

 * pytorch
 * numpy, scipy, matplotlib
 * PYSEQM (PYtorch-based Semi-Empirical Quantum Mechanics, https://github.com/lanl/PYSEQM )
   A copy of this python module is provided in the folder PYSEQM
 * tqdm

---------------
Getting Started
---------------
The package is installed by running

.. code-block:: bash

   $ pip install -e .
   
in the top directory.

Since the scan is calculated in parallel, the code runs significantly faster when a GPU is available.

-------
Example
-------
The input should be an xyz-file with the geometry of the monomer lying in the XY plane.
To scan the singlet fission rate (approximated by |T_RP|^2) as a function of the
parallel displacement run

.. code-block::
   
   $ scan_sf_rate.py ethene_d2h.xyz --approximation='diabatic'

or

.. code-block::

   $ scan_sf_rate.py ethene_d2h.xyz --approximation='non-adiabatic'
   
The scans are stored in .npz files which can be plotted with

.. code-block::

   $ plot_sf_rate.py ethene_d2h_SF-diabatic.npz

or

.. code-block::

   $ plot_sf_rate.py ethene_d2h_SF-non-adiabatic.npz 
   
The four frontier monomer orbitals hA,hB, lA,lB for the dimer geometry which is estimated to
have the largest singlet fission rate, are saved to `monomer_orbitals_max_rate.molden`.

The geometries needed to reproduce the figures in the article [Doping]_ can be found
in the folder `example_geometries/`.

-------
Authors
-------
Alexander Humeniuk
Anurag Singh

----------
References
----------

.. [Doping] Anurag Singh, Alexander Humeniuk, Merle Röhr,
   *Optimal Molecular Packing for Singlet Fission in Perylene Doped with Boron and Nitrogen*
   in preparation

.. [Simple] E. Buchanan, Z. Havlas, J. Michl,
   *Singlet fission: Optimization of chromophore dimer geometry*
   Advances in Quantum Chemistry (2017), 75, 175-227.
	
.. [PYSEQM] Zhou, Guoqing, et al.
    *Graphics processing unit-accelerated semiempirical Born Oppenheimer molecular dynamics using PyTorch.*
    Journal of Chemical Theory and Computation 16.8 (2020): 4951-4962.
