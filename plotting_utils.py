#dumping useful functions to adjust matplotlib figures that I use frequently
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#some default font stuff to use
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["text.usetex"] = True

#gets the grid closest to the desired aspect ratio
#TODO: There's a better way if I'm not lazy
def grid_by_aspect(n,aspect=4./3.):
    nrows = 1
    ncols = 1
    for ii in range(1,n+1):
        if ii > nrows * ncols:
            ratio_test = ncols/nrows
            if ratio_test < aspect:
                ncols += 1
            else:
                nrows += 1

    return nrows,ncols

#for gridspec plots with shared axes, adjust ticks & axis labels to only show up on the edges
def fix_ticks(fig,sharex=True,sharey=True):
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        if ss is None:
            continue
        if not ss.is_last_row() and sharex:
            ax.tick_params(bottom=False,labelbottom=False)
            ax.set_xlabel('')
        if not ss.is_first_col() and sharey:
            ax.tick_params(left=False,labelleft=False)
            ax.set_ylabel('')

def find_gridspec(n_plots,aspect=4/3.):
    nrows = 1
    ncols = 1
    for ii in range(1,n_plots+1):
        ratio_test = ncols/nrows
        if ii > nrows * ncols:
            if ratio_test < aspect:
                ncols += 1
            else:
                nrows += 1
    return nrows,ncols

def register_eor_cmaps():
    my_cmap_bt = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap_diff',
                                                                     [(1.0, 'lightsteelblue'),
                                                                      (0.9, 'cornflowerblue'),
                                                                       (0.7, 'royalblue'),
                                                                       (0.5, 'black'),
                                                                       (0.3, 'orangered'),
                                                                       (0.1, 'coral'),
                                                                       (0.0, 'wheat')][::-1])
    try:
        matplotlib.colormaps.register(cmap=my_cmap_bt)
    except ValueError:
        matplotlib.colormaps.unregister('my_cmap_diff')
        matplotlib.colormaps.register(cmap=my_cmap_bt)
    
    EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap_eor',
                                                                     [(0, 'white'),
                                                                      (0.33, 'yellow'),
                                                                      (0.5, 'orange'),
                                                                      (0.68, 'red'),
                                                                      (0.833, 'black'),
                                                                      (0.87, 'blue'),
                                                                      (1, 'cyan')])
    
    try:
        matplotlib.colormaps.register(cmap=EoR_colour)
    except ValueError:
        matplotlib.colormaps.unregister('my_cmap_eor')
        matplotlib.colormaps.register(cmap=EoR_colour)

    return my_cmap_bt,EoR_colour

plot_specs = {
    "density" : {
        "cmap" : 'viridis',
        "vmin" : -1,
        "vmax" : 1,
        "lognorm" : False,
        "clabel" : r'$\delta$',
    },
    "halo_mass" : {
        "cmap" : 'Purples',
        "vmin" : 1e8,
        "vmax" : 1e11,
        "lognorm" : True,
        "clabel" : r'$\rho_{halo} (M_\odot cMpc^{-3})$',
    },
    "n_ion" : {
        "cmap" : 'Reds',
        "vmin" : 1e-3,
        "vmax" : 1e1,
        # "vmin" : 1e7,
        # "vmax" : 1e12,
        "lognorm" : True,
        "clabel" : r'$\rho_\gamma (cMpc^{-3})$',
    },
    "halo_stars" : {
        "cmap" : 'Reds',
        "vmin" : 1e5,
        "vmax" : 1e10,
        "lognorm" : True,
        "clabel" : r'$\rho_* (M_\odot cMpc^{-3})$',
    },
    "halo_stars_mini" : {
        "cmap" : 'Reds',
        "vmin" : 1e4,
        "vmax" : 1e7,
        "lognorm" : True,
        "clabel" : r'$\rho_* (M_\odot cMpc^{-3})$',
    },
    "halo_sfr" : {
        "cmap" : 'Blues',
        "vmin" : 1e-12,
        "vmax" : 1e-7,
        "lognorm" : True,
        "clabel" : r'SFRD $(M_\odot s^{-1} cMpc^{-3})$',
    },
    "halo_sfr_mini" : {
        "cmap" : 'Blues',
        "vmin" : 1e-13,
        "vmax" : 1e-9,
        "lognorm" : True,
        "clabel" : r'SFRD $(M_\odot s^{-1} cMpc^{-3})$',
    },
    "whalo_sfr" : {
        "cmap" : 'Blues',
        "vmin" : 1e-11,
        "vmax" : 1e-8,
        "lognorm" : True,
        "clabel" : r'SFRD $(M_\odot s^{-1} cMpc^{-3})$',
    },
    "Tk_box" : {
        "cmap" : 'cividis',
        "vmin" : 1,
        "vmax" : 100,
        "lognorm" : False,
        "clabel" : r'$T_K$',
    },
    "Ts_box" : {
        "cmap" : 'plasma',
        "vmin" : 0,
        "vmax" : 100,
        "lognorm" : False,
        "clabel" : r'$T_s$',
    },
    "xH_box" : {
        "cmap" : 'magma',
        "vmin" : 0,
        "vmax" : 1,
        "lognorm" : False,
        "clabel" : r'$x_H$',
    },
    "brightness_temp" : {
        "cmap" : 'my_cmap_eor',
        "vmin" : -150,
        "vmax" : 30,
        "lognorm" : False,
        "clabel" : r'$T_{b,21} (\mathrm{mK})$',
    },
    "brightness_temp_diff" : {
        "cmap" : 'my_cmap_diff',
        "vmin" : -50,
        "vmax" : 50,
        "lognorm" : False,
        "clabel" : r'$T_{b,21} (\mathrm{mK})$',
    },
    "CII_box" : {
        "cmap" : 'cividis',
        "vmin" : 1e-4,
        "vmax" : 1e1,
        "lognorm" : True,
        "clabel" : r'$L_\mathrm{CII} (\mathrm{kJy sr}^{-1})$'
    },
    "Gamma12_box" : {
        "cmap" : 'viridis',
        "vmin" : 1e-2,
        "vmax" : 1e2,
        "lognorm" : True,
        "clabel" : r'$\Gamma_{12} (s-1)$'
    },
    "dNrec_box" : {
        "cmap" : 'gist_rainbow',
        "vmin" : 1e-2,
        "vmax" : 1e1,
        "lognorm" : True,
        "clabel" : r'$N_\mathrm{rec}$'
    },
    "count" : {
        "cmap" : 'magma',
        "vmin" : 1e0,
        "vmax" : 1e3,
        "lognorm" : True,
        "clabel" : r'$N_\mathrm{gal}$'
    },
    "halo_xray" : {
        "cmap" : 'magma',
        "vmin" : 1e-2,
        "vmax" : 1e3,
        "lognorm" : True,
        "clabel" : r'$L_X (10^{38} \mathrm{erg \, s}^{-1} \, \mathrm{Mpc}^{-3})$'
    }
}

def find_pspec(kind,obj):
    #if there's an exact match, take it
    if kind in plot_specs.keys():
        return plot_specs[kind]
    
    #otherwise look for a substring in keys
    for key in plot_specs.keys():
        if key in kind:
            return plot_specs[key]

    #if all else fails, make a fake pspec that seems reasonable
    arr = getattr(obj,kind)
    return {
        "cmap" : 'plasma',
        "vmin" : np.percentile(arr,2.5),
        "vmax" : np.percentile(arr,97.5),
        "lognorm" : np.all(arr > 0) & (np.fabs(arr.max()) > np.fabs(arr.min*100)), #probably some better way
        "clabel" : f'{kind}'
    }
    
