#!/usr/bin/env python
"""
show 2D scan with maxima marked by dots
"""
import argparse
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def find_local_maxima(x,y, z, n=2):
    """
    find local maxima of the function z = f(x,y) on a rectangular grid

    Parameters
    ----------
    x  :  ndarray (nx,ny)
      x[i,j] is the x-position of the point (i,j)
    y  :  ndarray (nx,ny)
      y[i,j] is the y-position of the point (i,j)
    z  :  ndarray (nx,ny)
      z[i,j] is the function value f(x[i,j], y[i,j])

    Returns
    -------
    xs, ys : ndarrays (n,)
      x- and y-positions of the n largest local maxima
    zs     : narray (n,)
      function values at the n largest local maxima
    """
    zmax = filters.maximum_filter(z, size=20)
    # 
    maxima = (z == zmax)
    # identify connected areas 
    label, num_features = ndimage.label(maxima)
    # generate a list of slices for the labeled features
    slices = ndimage.find_objects(label)

    # x- and y-positions at which the maxima occur
    xs = []
    ys = []
    # z-values at the maxima
    zs = []
    
    for slice_i, slice_j in slices:
        i = (slice_i.start + slice_i.stop - 1)//2
        j = (slice_j.start + slice_j.stop - 1)//2
        xs.append( x[i,j] )
        ys.append( y[i,j] )
        zs.append( z[i,j] )

    # convert to numpy arrays
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    
    # sort local maxima by z value and take the largest n ones
    idx = np.argsort(zs)[-n:]

    return xs[idx], ys[idx], zs[idx]


import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('axes', labelsize=16)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plot 2D scan of |T_RP|^2 and identify local maxima')
parser.add_argument('npz_file',
                    type=str, metavar='scan.npz',
                    help='npz file generated by the script scan_sf_rate.py')
args = parser.parse_args()

plt.xlabel(r"$\Delta X / \AA$")
plt.ylabel(r"$\Delta Y / \AA$")

data = dict(np.load(args.npz_file))

nx = int(np.sqrt(len(data['dx'])))
ny = nx
dX = np.reshape(data['dx'], (nx,ny))
dY = np.reshape(data['dy'], (nx,ny))

rates = np.reshape(data['rates'], (nx,ny))

xs,ys,zs = find_local_maxima(dX,dY, rates, n=2)

# show 2D scan
cmap = plt.get_cmap('jet')
plt.pcolormesh(dX, dY, rates, cmap=cmap, shading='nearest')

# put little dots at the local maxima
plt.plot(xs, ys, "o", color="white", label="local maxima")
plt.legend()

plt.colorbar()
        
plt.tight_layout()
plt.savefig(args.npz_file.replace(".npz", ".svg"))
plt.savefig(args.npz_file.replace(".npz", ".png"), dpi=300)
plt.show()
