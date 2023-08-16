import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import datetime

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
inferno_r_cmap = cm.get_cmap('inferno_r')
# my_cmap.set_under('w') # don't seem to work
xr = np.linspace(0, 1, 256)
inferno_r_cmap_listed = inferno_r_cmap(xr)
inferno_r_whitebg_cmap_listed = np.vstack((np.array([np.ones(4)+(inferno_r_cmap_listed[0]-np.ones(4))*x for x in np.linspace(0,1,int(256/8))]),inferno_r_cmap_listed[:-int(256/16)]))
inferno_r_whitebg_cmap = ListedColormap(inferno_r_whitebg_cmap_listed)
cmap = inferno_r_whitebg_cmap
def full_path(path):
    """
    From C. Mayes
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))

"""UTC to ISO 8601 with Local TimeZone information without microsecond"""
def isotime():
    """
    From C. Mayes.
    Get time stamp for filename
    :return:
    """
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat().replace(':','_')
def plot_surface(x, y, z):
    """
    :param x: 1D x grid (Nx,)
    :param y: 1D y grid (Ny, )
    :param z: 2D z data (Nx, Ny)
    :return:
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X, Y = np.meshgrid(x, y, indexing = 'ij')


    # Plot the surface.
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_2D_contour(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(z, extent=(min(x)*1e6, max(x)*1e6, min(y)*1e6, max(y)*1e6), origin='lower',  cmap=cmap)
    plt.xlabel('y ($\mu m$)')
    plt.ylabel('x ($\mu m$)')
    plt.show()