import numpy as np

import py21cmfast as p21c
from astropy import units as U, constants as C
from astropy.cosmology import z_at_value
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize,LogNorm
from matplotlib import gridspec

from .plotting_utils import plot_specs as pspecs, fix_ticks, register_eor_cmaps, find_pspec, grid_by_aspect
from .analysis_funcs import match_global_function, get_lc_powerspectra, get_props_from_halofield
from .postprocessing import xray_bg, xray_xps, cii_xps, make_cii_map, gal_xps, make_gal_lc, make_cii_coev
from .spec_conversions import sfr_to_Muv
from .io import find_closest_box, read_cv_ignoreparams, read_lc_ignoreparams

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def summary_plot(lc_file,cv_file,hf_file,outname):
    fig,axs = plt.subplots(2,2,figsize=(8,8),layout="constrained",squeeze=False)
    axs = axs.flatten()
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0,
                                wspace=0.0)
    
    #UVLF PLOT
    uv_lf_ax([hf_file,],['',],axs[0])
    axs[0].text(0.05,0.95,r'$z=8$',transform=axs[0].transAxes,verticalalignment='top',horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    lc = p21c.LightCone.read(lc_file,safe=False)
    setattr(lc,"CII_Box",make_cii_map(lc)[0])

    #21cm PS FIXED Z
    powerspec_plot([lc,],None,None,z_out=(0.5,),z_type='XHI',axs=[axs[1],])
    
    #21cm PS FIXED K
    largescale_powerplot([lc,],None,None,k_target=(0.1,),axs=[axs[2],])

    #CII X PLOT
    k_arr, power_arr, z_ps = get_lc_powerspectra([lc,],z_list=[12,],kind='brightness_temp',kind2='CII_Box',
                                                subtract_mean=[False,True],divide_mean=[False,True],
                                                n_psbins=20,z_type='redshift')
    axs[3].set_xlim(k_arr[0,:,1:].min(),k_arr.max())
    axs[3].set_ylabel(r'$\left< \delta T_{b,21} \delta_{CII} \right> \frac{k^3}{2\pi}$ (mK)')
    axs[3].set_xlabel(r'$k (\mathrm{Mpc}^{-1})$')
    axs[3].plot(k_arr[0,0,:],power_arr[0,0,:])
    
    axs[3].set_yscale('symlog',linthresh=2e-1)
    axs[3].set_xscale('log')
    axs[3].grid()

    fix_ticks(fig,sharex=False,sharey=False)

    fig.savefig(outname)

def uv_lf_ax(hf_ax,names,ax,pth=True):
    ax.set_xlim([-15,-24.5])
    ax.set_xlabel(r"$\mathrm{M_{UV}}$")
    ax.set_ylabel(r"$\phi \mathrm{(mag^{-1} Mpc^{-3})}$")
    ax.set_yscale('log')
    ax.grid()
    
    Muvbins = np.linspace(-30,-15,num=40)
    muv_cen = (Muvbins[:-1] + Muvbins[1:])/2
    muv_wid = Muvbins[1:] - Muvbins[:-1]
    #get the theoretical UVLF from the inputs of the first halofield in the group
    if pth:
        hf_example = p21c.PerturbHaloField.from_file(hf_ax[0],arrays_to_load=())
    else:
        hf_example = p21c.HaloField.from_file(hf_ax[0],arrays_to_load=())

    # Muvbins_inp,_,uvlf_from_inputs = p21c.compute_luminosity_function(redshifts=[hf_example.redshift,],
    #                                                     user_params=hf_example.user_params,
    #                                                     cosmo_params=hf_example.cosmo_params,
    #                                                     astro_params=hf_example.astro_params,
    #                                                     flag_options=hf_example.flag_options,
    #                                                     nbins=45,)
    # uvlf_from_inputs = 10**(uvlf_from_inputs.flatten()[::-1])
    # ax.plot(Muvbins_inp,uvlf_from_inputs,'k:')

    volume = (hf_example.user_params.BOX_LEN)**3
    one_halo = 1 / volume / muv_wid
    ax.plot(muv_cen,one_halo,color='xkcd:grey',linestyle='--')
    ax.set_ylim(one_halo[-1]/2,one_halo[0]*1e6)
    for j,hffile in enumerate(hf_ax):
        if pth:
            hf = p21c.PerturbHaloField.from_file(hffile,arrays_to_load=None)
        else:
            hf = p21c.HaloField.from_file(hffile,arrays_to_load=None)
        sfr, = get_props_from_halofield(hf)
        
        muv = sfr_to_Muv(sfr)
        uvlf,_ = np.histogram(muv,Muvbins)
        volume = (hf.user_params.BOX_LEN)**3
        ax.plot(muv_cen,uvlf / muv_wid / volume,color=f"C{j:02d}",linestyle='-',label=names[j])


def uv_lf_plot(halofield_list,names,titles,outname,pth=True):
    '''
    Make a UV Luminosity function plot

    Arguments:
    halofield_list: list(list(str))
    The list of filenames of the HaloField or PerturbHaloField objects. UVLFs will be grouped on axes by
    the first axis (a common operating mode is the first axis to be groups of the same redshift,
    and the second having parameter variation)

    names: list(str)
    list of names in the legend, must be the same size as the second axis of halofield_list

    titles: list(str)
    list of titles of the subplots, must be the same size as the first axis of halofield_list

    outname: Path
    filename to save the plot

    Returns: None
    '''
    nrows,ncols = grid_by_aspect(len(halofield_list),aspect=3.)
    fig,axs = plt.subplots(nrows,ncols,figsize=(8,6*nrows/ncols),layout="constrained",squeeze=False)
    axs = axs.flatten()
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0,
                                wspace=0.0)
    
    for i,ax in enumerate(axs):
        hf_ax = halofield_list[i]
        uv_lf_ax(hf_ax,names,ax,pth)
        if titles[i]:
            ax.text(0.05,0.95,titles[i],transform=axs[i].transAxes,verticalalignment='top',horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    axs[0].legend()
    fig.savefig(outname)

def gal_colocation_plot(coev_list,halofield_list,names,outname,cv_kinds=None,hf_kinds=None,n_largest=300,slice_index=0,slice_width=1):
    register_eor_cmaps()
    nrows,ncols = grid_by_aspect(len(coev_list),aspect=4.1)
    fig,axs = plt.subplots(nrows,ncols,figsize=(8,8*nrows/ncols),layout='constrained',sharex=True,sharey=True)
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0, wspace=0.0)
    axs = axs.flatten()

    if cv_kinds is None:
        cv_kinds = 'brightness_temp'
    if hf_kinds is None:
        hf_kinds = 'halo_masses'
    
    one_cv = False
    if isinstance(cv_kinds,str):
        one_cv = True
        cv_kinds = [cv_kinds,]*len(coev_list)
        
    if isinstance(hf_kinds,str):
        hf_kinds = [hf_kinds,]*len(halofield_list)

    for i,(coevf,halofieldf,cvk,hfk) in enumerate(zip(coev_list,halofield_list,cv_kinds,hf_kinds)):
        coev = p21c.Coeval.read(coevf,safe=False)

        if 'CII_box' == cvk:
            setattr(coev,"CII_box",make_cii_coev(coev))

        halofield = p21c.PerturbHaloField.from_file(halofieldf,arrays_to_load=None)

        slmin = slice_index - ((slice_width-1)//2)
        slmax = slice_index + (slice_width//2 + 1)
        if slmin < 0 or slmax >= coev.user_params.HII_DIM:
            raise NotImplementedError("no wrapping yet")
        cv_slice = getattr(coev,cvk)[:,:,slmin:slmax].mean(axis=-1)
        #select halos of the right slice
        hf_pos_sel = (halofield.halo_coords[:,2] >= slmin) & (halofield.halo_coords[:,2] < slmax)

        if 'Muv' == hfk:
            sfr, = get_props_from_halofield(halofield,None,None,sel=hf_pos_sel,kinds=['sfr'])
            hf_prop = sfr_to_Muv(sfr)
        else:
            hf_prop, = get_props_from_halofield(halofield,None,None,sel=hf_pos_sel,kinds=[hfk,])
        hf_pos = halofield.halo_coords[hf_pos_sel,:] * coev.user_params.BOX_LEN / coev.user_params.HII_DIM
        
        spec = find_pspec(cvk,coev)
        #NOTE: does not show slice width yet
        p21c.plotting.coeval_sliceplot(coev,cmap=spec['cmap'],kind=cvk,fig=fig,ax=axs[i],aspect='equal',cbar=False
                                                    ,log=spec['lognorm'],cbar_label=spec['clabel'],vmin=spec['vmin']
                                                    ,vmax=spec['vmax'],slice_index=slice_index)
        
        #select largest N halos
        if hfk == "Muv":
            #points and labels will be overwritten by the same thing
            points = []
            sel_bins = np.array([-18,-19,-20,-21],dtype=int)
            marker_list = ['o','s','v','X']
            labels = [rf'$M_{{uv}} \in [{{{sel_bins[j]}}},{{{sel_bins[j+1]}}}]$' for j in range(sel_bins.size-1)]
            for j in range(len(sel_bins) - 1):
                sel = (hf_prop < sel_bins[j]) & (hf_prop > sel_bins[j+1])
                p = axs[i].scatter(hf_pos[sel,0],hf_pos[sel,1],marker=marker_list[j],c='none',edgecolors='xkcd:neon green',
                               linewidths=0.7,s=10)
                points.append(p)
        else:
            k_part = hf_prop.size - n_largest - 1
            hf_lim_sel = np.argpartition(hf_prop,k_part)[k_part:]
            hf_prop = hf_prop[hf_lim_sel]
            hf_pos = hf_pos[hf_lim_sel,:]

            msize_max = 20
            msize_min = 1
            msize_p = np.log10(hf_prop/hf_prop.min())/np.log10(hf_prop.max()/hf_prop.min())
            gal_marker_size = msize_min + (msize_max - msize_min)*msize_p
            axs[i].scatter(hf_pos[:,0],hf_pos[:,1],c='none',edgecolors='xkcd:neon green',linewidths=0.7,s=gal_marker_size)
            
        axs[i].text(0.05,0.95,names[i],transform=axs[i].transAxes,verticalalignment='top',horizontalalignment='left',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    fix_ticks(fig)
    if one_cv:
        cb = plt.colorbar(axs[0].get_images()[0],ax=axs)
        cb.ax.set_xlabel(spec['clabel'],fontsize=8)
    if 'Muv' in hf_kinds:
        fig.legend(points,labels,loc='upper center', bbox_to_anchor=(0.5,0.98), ncol=sel_bins.size-1, fancybox=True)
    fig.savefig(outname)

def full_cross_plot(lc_list,names=('lc1','lc2'),z_out=12.,kind='brightness_temp',kind2='CII_box',n_psbins=20,outname=None,vertical=False):
    ncols = 1 if vertical else 3
    nrows = 3 if vertical else 1
    plot_dims = (4,8) if vertical else (8,2.5)
    fig,axs = plt.subplots(nrows,ncols,figsize=plot_dims,layout='constrained',sharex=True)
    fig.get_layout_engine().set(w_pad=3 / 72, h_pad=3 / 72, hspace=0.0, wspace=0.0)

    #auto 1
    z_out = np.array([z_out,])
    k_arr, power_arr, z_ps = get_lc_powerspectra(lc_list,z_list=z_out,kind=kind,
                                                subtract_mean=[False,False],divide_mean=[False,False],
                                                n_psbins=n_psbins,z_type='redshift')
    for j in range(len(lc_list)):
        axs[0].set_xlim(k_arr[0,:,1:].min(),k_arr.max())
        axs[0].set_ylabel(r'$\Delta_{21} \overline{dT}_{b,21} (mK^2)$')
        axs[0].set_xlabel(r'$k (\mathrm{Mpc}^{-1})$')
        axs[0].loglog(k_arr[j,0,:],power_arr[j,0,:],label=names[j])
        axs[0].grid()
        axs[0].text(0.05,0.95,f'z={z_ps[0,0]:.1f}',transform=axs[0].transAxes,verticalalignment='top',horizontalalignment='left',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        if names:
            axs[0].legend(loc='lower right')

    #auto 2
    k_arr, power_arr, z_ps = get_lc_powerspectra(lc_list,z_list=z_out,kind=kind2,
                                                subtract_mean=[True,True],divide_mean=[True,True],
                                                n_psbins=n_psbins,z_type='redshift')
    for j in range(len(lc_list)):
        axs[1].set_xlim(k_arr[0,:,1:].min(),k_arr.max())
        axs[1].set_ylabel(r'$\Delta_{CII}$')
        axs[1].set_xlabel(r'$k (\mathrm{Mpc}^{-1})$')
        axs[1].loglog(k_arr[j,0,:],power_arr[j,0,:],label=names[j])
        axs[1].grid()

    #cross
    k_arr, power_arr, z_ps = get_lc_powerspectra(lc_list,z_list=z_out,kind=kind,kind2=kind2,
                                                subtract_mean=[False,True],divide_mean=[False,True],
                                                n_psbins=n_psbins,z_type='redshift')
    for j in range(len(lc_list)):
        axs[2].set_xlim(k_arr[0,:,1:].min(),k_arr.max())
        axs[2].set_ylabel(r'$\left< \delta T_{b,21} \delta_{CII} \right> \frac{k^3}{2\pi}$ (mK)')
        axs[2].set_xlabel(r'$k (\mathrm{Mpc}^{-1})$')
        axs[2].plot(k_arr[j,0,:],power_arr[j,0,:],label=names[j])
        
        axs[2].set_yscale('symlog',linthresh=2e-1)
        axs[2].set_xscale('log')
        axs[2].grid()

    fix_ticks(fig,sharex=True,sharey=False)

    fig.savefig(outname)
    


def ionisation_state_plot(cvfiles,outname):
    fig,ax = plt.subplots(2,len(cvfiles),figsize=(16.,12./len(cvfiles)*2.),layout='constrained',squeeze=False)
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0, wspace=0.0)
    
    bins_dens = np.linspace(-1,5,num=100)
    bins_nion = np.logspace(-3,2,num=101)

    for i,cvfile in enumerate(cvfiles):
        logger.info(f'starting {i+1} of {len(cvfiles)}: {cvfile}')
        cv = p21c.Coeval.read(cvfile,safe=False)
        # cv = read_cv_ignoreparams(cvfile)
        x_vals = cv.density.flatten()
        z_vals = cv.xH_box.flatten()
        norm = Normalize(vmin=0,vmax=1)
        ax[0,i].set_xlabel('delta')
        ax[0,i].set_yscale('log')
        ax[0,i].plot(bins_dens,(1+bins_dens),'k:')
        ax[0,i].set_ylim(1e-3,1e2)
        ax[0,i].grid()
        bins = [bins_dens,bins_nion]
        ax[0,i].set_ylabel('n_ion/rhocrit')
        #if we have no halos we do a dens vs xH histogram instead of a binned xH in nion and density
        rhocrit = (cv.cosmo_params.cosmo.critical_density(0)).to('M_sun Mpc-3').value
        if hasattr(cv,"n_ion"):
            y_vals = (cv.n_ion/rhocrit/cv.cosmo_params.OMb).flatten()            
        else:
            y_vals = (
                cv.Fcoll.flatten()
                * (10**cv.astro_params.F_STAR10)
                * (10**cv.astro_params.F_ESC10)
                * cv.global_params['Pop2_ion']
                * (1 + x_vals)
            )

        hist,_,_,_ = binned_statistic_2d(x_vals,y_vals,values=z_vals,bins=bins,statistic='mean')
        im = ax[0,i].pcolormesh(bins[0],bins[1],hist.T,shading='flat',norm=norm)
        
        ax[0,i].set_title(cvfile.split('/')[-1])
        dim = np.linspace(0,cv.user_params.BOX_LEN,num=cv.user_params.HII_DIM+1)
        im2 = ax[1,i].pcolormesh(dim,dim,cv.xH_box[:,:,0],shading='flat',norm=Normalize(vmin=0,vmax=1),cmap='plasma')

    plt.colorbar(im,ax=ax[0,:])
    plt.colorbar(im2,ax=ax[1,:])

    fig.savefig(outname)


#Scatter plots of
def field_comparison_plot_grid(cvfiles,outname,kind='xH_box',logscale=False):
    plot_dim = len(cvfiles)-1
    fig,ax = plt.subplots(plot_dim,plot_dim,figsize=(16.,12),layout='constrained',squeeze=False,sharex=True,sharey=True)
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0, wspace=0.0)
    
    #read in the data
    boxes = []
    dens_boxes = []
    for i,cvfile in enumerate(cvfiles):
        logger.info(f'reading {cvfile} ({i+1} of {len(cvfiles)})')
        cv = p21c.Coeval.read(cvfile,safe=False)
        boxes += [getattr(cv,kind),]
        dens_boxes += [cv.density,]

    for i,box in enumerate(boxes):
        for j,box2 in enumerate(boxes):
            if j >= i:
                continue
            logger.info(f'starting {i} {j}')
            im = ax[i-1,j].scatter(box2,box,s=1,c=dens_boxes[i],norm=Normalize(vmin=0,vmax=1))

            if logscale:
                ax[i-1,j].set_yscale('log')
                ax[i-1,j].set_xscale('log')
            ax[i-1,j].grid()

            if j == 0:
                ax[i-1,j].set_ylabel(f'{cvfiles[j]}')
            if i == 0:
                ax[i-1,j].set_title(f'{cvfiles[i]}')
            
    
    fig.colorbar(im,ax=ax)
    fig.savefig(outname)

def field_comparison_plot(cvfiles,outname,kind='xH_box',logx=False):
    fig,ax = plt.subplots(1,1,figsize=(16.,12),layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0, wspace=0.0)
    
    #read in the data
    boxes = []
    dens_boxes = []
    for i,cvfile in enumerate(cvfiles):
        logger.info(f'reading {cvfile} ({i+1} of {len(cvfiles)})')
        cv = p21c.Coeval.read(cvfile,safe=False)
        boxes += [getattr(cv,kind),]
        dens_boxes += [cv.density,]

    for i,box in enumerate(boxes):
        logger.info(f'starting {i+1} of {len(cvfiles)}')
        ax.scatter(dens_boxes[i],box,s=1,label=cvfiles[i])
    
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('delta')
    ax.set_ylabel(f'{kind}')
    ax.grid()
    fig.savefig(outname)


def global_integral_plot(lcfiles,fields,outname,names,zmax=20):
    fig,ax = plt.subplots(2,len(fields),figsize=(16,12/len(fields)*2),layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0,
                            wspace=0.0)
    
    for j,lcfile in enumerate(lcfiles):
        lc = p21c.LightCone.read(lcfile,safe=False)
        if not hasattr(lc,"global_n_ion"):
            logger.info(f"{lcfile} has no halos, continuing...")
            continue
        zmax = float(zmax) if zmax is not None else lc.node_redshifts[-1]
        plot_idx = lc.node_redshifts < zmax
        plot_z = lc.node_redshifts[plot_idx]

        if lc.flag_options.USE_MINI_HALOS:
            Mmin = lc.global_params['M_MIN_INTEGRAL']
        else:
            Mmin = 10**lc.astro_params.M_TURN / 50.
        # Mmin = lc.global_params["SAMPLER_MIN_MASS"]
        Mmax = lc.global_params["M_MAX_INTEGRAL"]
        kwargs_expected = {'Mmin' : Mmin,
                'Mmax' : Mmax,
                'lnMmin' : np.log(Mmin),
                'lnMmax' : np.log(Mmax)}

        #only do expected globals for first LC (assume same params)
        if j==0:
            expected_global = match_global_function(fields,lc,**kwargs_expected)[:,plot_idx]
            [ax[0,i].plot(plot_z,expected_global[i],'r:',linewidth=5) for i,field in enumerate(fields)]

        for i,field in enumerate(fields):
            field_lc = 'global_' + field
            if field_lc.endswith("_box"):
                field_lc = field_lc[:-4]

            f_array = getattr(lc,field_lc)[plot_idx]
            # for xH_box we want special treatment
            if field == 'xH_box':
                f_array = 1-f_array
                if "n_ion" in fields:
                    rhocrit = (lc.cosmo_params.cosmo.critical_density(0)).to('M_sun Mpc-3').value
                    f_array2 = getattr(lc,"global_n_ion")[plot_idx] / rhocrit / lc.cosmo_params.OMb
                    ax[0,i].plot(plot_z,f_array2,color=f'C{j:02d}',linestyle=':')

            ax[0,i].plot(plot_z,f_array,color=f'C{j:02d}',label=names[j] if names else None)
            ax[1,i].plot(plot_z,(f_array/expected_global[i]),color=f'C{j:02d}')

    for i,field in enumerate(fields):
        ax[0,i].set_yscale('log')
        ax[0,i].grid()
        ax[0,i].set_xlabel('redshift')
        ax[0,i].set_ylabel(f'{pspecs[field]["clabel"]}')

        ax[1,i].set_yscale('log')
        ax[1,i].grid()
        ax[1,i].set_xlabel('redshift')
        ax[1,i].set_ylabel(f'Ratio')
        ax[1,i].set_ylim([0.5,2])

    if names:
        ax[0,0].legend()
    fig.savefig(outname)


def plot_xrb(titles,noise_x=1e-16,res_x_arcsec=5,noise_flag=True):
    #setup inputs and parameters
    lightcones = [p21c.LightCone.read(title,safe=False) for title in titles]

    fig, axs = plt.subplots(1 + noise_flag,len(lightcones), figsize=(12,(1+noise_flag)*12/len(lightcones)),squeeze=False)
    axs[0,0].set_ylabel('No Noise')
    if noise_flag:
        axs[1,0].set_ylabel('+ 1e-16 Gaussian Noise')
        random_field = np.random.normal(loc=0,scale=noise_x,size=(lightcones[0].user_params.HII_DIM,lightcones[0].user_params.HII_DIM))

    for i,lc in enumerate(lightcones):
        #find the minimum angular diameter distance, <ASSUME THIS IS CONSTANT?????>
        min_z = lc.lightcone_redshifts.min()
        d_a = lc.cosmo_params.cosmo.angular_diameter_distance(min_z)
        ang_size = ((lc.user_params.BOX_LEN/lc.user_params.HII_DIM) / (1+min_z) * U.Mpc / d_a) * U.rad
        res_rad = (res_x_arcsec * U.arcsec).to('rad')
        gauss_sigma = (res_rad/ang_size).to('').value
        # print(f'sigma = {gauss_sigma}')

        x_lc,err = xray_bg(lc)
        logger.info(f'DONE {titles[i]}')

        x_lc = gaussian_filter(x_lc,sigma=gauss_sigma,mode='wrap')

        im = axs[0,i].imshow(x_lc,cmap='viridis',norm=Normalize(vmin=0,vmax=3e-16),interpolation='nearest',extent=[0,200,0,200])
        if noise_flag:
            im2 = axs[1,i].imshow(x_lc + random_field,cmap='viridis',norm=Normalize(vmin=0,vmax=3e-16),interpolation='nearest',extent=[0,200,0,200])
        axs[0,i].set_title(titles[i])

    fig.subplots_adjust(top=0.95,bottom=0.1,left=0.1,right=0.9)
    plt.colorbar(im,ax=axs[:,:],label=r'XRB(erg cm$^{-2}$ s$^{-1}$)')
    fig.savefig('./plots/xrb.png')


def cell_history_plot(lcfile,cache):    
    p21c.config['direc'] = cache
    lc = p21c.LightCone.read(lcfile,safe=False)

    z_targets = np.linspace(6,12,num=20)

    hboxes,z_matches = find_closest_box(z_targets,p21c.HaloBox,lc)

    lo_dim = lc.user_params.HII_DIM
    sel = np.random.randint(lo_dim,size=(16,3))

    fig,axs = plt.subplots(nrows=1,ncols=3)

    mean_sfr = np.array([hbox['halo_sfr'].mean for hbox in hboxes])
    erru_sfr = np.array([np.percentile(hbox['halo_sfr'],84) for hbox in hboxes])
    errl_sfr = np.array([np.percentile(hbox['halo_sfr'],16) for hbox in hboxes])
    cell_sfr = np.array([hbox['halo_sfr'][sel] for hbox in hboxes])
    print(cell_sfr.shape)

    mean_star = np.array([hbox['stellar_mass'].mean for hbox in hboxes])
    erru_star = np.array([np.percentile(hbox['stellar_mass'],84) for hbox in hboxes])
    errl_star = np.array([np.percentile(hbox['stellar_mass'],16) for hbox in hboxes])
    cell_star = np.array([hbox['stellar_mass'][sel] for hbox in hboxes])

    mean_mass = np.array([hbox['halo_mass'].mean for hbox in hboxes])
    erru_mass = np.array([np.percentile(hbox['halo_mass'],84) for hbox in hboxes])
    errl_mass = np.array([np.percentile(hbox['halo_mass'],16) for hbox in hboxes])
    cell_mass = np.array([hbox['halo_mass'][sel] for hbox in hboxes])

    axs[0].fill_between(z_matches,errl_mass,erru_mass,color='xkcd:light grey',linewidth=3)
    axs[0].semilogy(z_matches,mean_mass,'k-',linewidth=3)
    axs[0].semilogy(z_matches,cell_mass,'k-',linewidth=0.5)

    axs[1].fill_between(z_matches,errl_mass,erru_mass,color='xkcd:light grey',linewidth=3)
    axs[1].semilogy(z_matches,mean_mass,'k-',linewidth=3)
    axs[1].semilogy(z_matches,cell_mass,'k-',linewidth=0.5)
    
    axs[2].fill_between(z_matches,errl_mass,erru_mass,color='xkcd:light grey',linewidth=3)
    axs[2].semilogy(z_matches,mean_mass,'k-',linewidth=3)
    axs[2].semilogy(z_matches,cell_mass,'k-',linewidth=0.5)

    fig.tight_layout()
    fig.savefig('cell_history.png')

def make_cii_xps_plot(lc,z_arr,k_arr,xps_arr,outname,z_range=(6,16),plot_diagnostic=False,**kwargs):
    """
    Make the CII cross-power plot 
    (2 lightcone slices, crosspower at series of z + scatter)

    lc : p21c.LightCone object or list of (Nlc,) LightCone objects
    z_arr : (Nz,) array of redshifts
    k_arr : (Nk,) array of scales
    xps_arr : (Nlc,Nz,Nk) OR (Nz,Nk) cross-power

    If lc is a list the cross-power plot is a mean + shaded scatter across all lightcones
    Otherwise it is simply the cross-power in the given lightcone
    """
    register_eor_cmaps()
    
    z_lc = lc.lightcone_redshifts

    ps_idx = np.argmin(np.fabs(z_lc[None,:] - z_arr[:,None]),axis=1)
    dist_convert = lc.user_params.BOX_LEN / lc.user_params.HII_DIM

    z_dim = z_lc.size
    x_dim = lc.user_params.HII_DIM

    gs = gridspec.GridSpec(ncols=2,nrows=2,width_ratios=(z_dim/x_dim/5,1.2),hspace=0.15,wspace=0.2)

    fig = plt.figure(figsize=(16,6.5))
    fig.subplots_adjust(top=0.99,bottom=0.1,left=0.1,right=0.95)

    ax = fig.add_subplot(gs[0,0])
    
    fig, ax = p21c.plotting.lightcone_sliceplot(lc,cmap="viridis",kind='CII_box',fig=fig,ax=ax,zticks='redshift',aspect='equal'
                                                ,log=True,cbar_label=r'$\mathrm{CII} (\mathrm{Jy sr}^{-1})$',vertical=False,cbar_horizontal=True
                                                ,vmin=1e-3,vmax=1e3,z_range=z_range,slice_index=24)
    ax.tick_params(bottom=False,labelbottom=False,top=True,labeltop=True)
    ax.xaxis.set_label_position('top') 
    
    # ax.set_title('CII Surface Brightness Density')
    [ax.axvline(p*dist_convert,color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline((p+lc.user_params.HII_DIM//2)*dist_convert,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline((p-lc.user_params.HII_DIM//2)*dist_convert,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    # limit_bt = np.percentile(lc.brightness_temp,90)
    limit_bt = 5.

    ax = fig.add_subplot(gs[1,0])
    
    fig, ax = p21c.plotting.lightcone_sliceplot(lc,cmap='my_cmap_diff',kind='brightness_temp_diff',fig=fig,ax=ax,zticks='redshift',aspect='equal'
                                                ,log=False,vertical=False,cbar_horizontal=True,cbar_label=r'$T_{b,21} (\mathrm{mK})$'
                                                ,vmin=-limit_bt,vmax=limit_bt,z_range=z_range,slice_index=24)
    ax.tick_params(bottom=False,labelbottom=False,top=True,labeltop=True)
    ax.xaxis.set_label_position('top') 
    
    [ax.axvline(p*dist_convert,color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline((p+lc.user_params.HII_DIM//2)*dist_convert,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline((p-lc.user_params.HII_DIM//2)*dist_convert,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    # ax.set_title('21cm Brightness Temperature')

    ax = fig.add_subplot(gs[:,1])
    logger.info(f'k_shape {k_arr.shape} xps shape {xps_arr.shape} (lc,z,k)')
    if len(xps_arr.shape) == 3:
        plot_means = np.mean(xps_arr,axis=0)
        plot_ul = np.percentile(xps_arr,97.5,axis=0)
        plot_ll = np.percentile(xps_arr,2.5,axis=0)
        logger.info(f'mean of means {plot_means.mean(axis=-1)}')
        [ax.plot(k_arr,plot_means[i,:],label=f'z={z}',color=f'C{i:02d}') for i,z in enumerate(z_arr)]
        [ax.fill_between(k_arr,plot_ll[i,:],plot_ul[i,:],color=f'C{i:02d}',alpha=0.2) for i,z in enumerate(z_arr)]
    elif len(xps_arr.shape) == 2:
        [ax.plot(k_arr,xps_arr[i,:],label=f'z={z}',color=f'C{i:02d}') for i,z in enumerate(z_arr)]
    else:
        raise ValueError(f"oops wrong dimensions {xps_arr.shape}")
    ax.set_yscale('symlog',linthresh=2e-1)
    ax.set_xscale('log')
    ax.set_xlabel('k (cMpc-1 h)')
    ax.set_ylabel(r'$\left< \delta T_{b,21} \delta_{CII} \right> \frac{k^3}{2\pi}$ (mK)')
    ax.set_title('1-D Cross Power Spectrum')
    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.93,bottom=0.1,left=0.05,right=0.95,hspace=0.1,wspace=0.1)

    fig.savefig(outname)

    if plot_diagnostic:
        fig,axs = plt.subplots(nrows=2,ncols=z_arr.size,figsize=(8,3))
        for j,idx in enumerate(ps_idx):
            p21c.plotting.lightcone_sliceplot(lc,cmap=pspecs['CII_box']['cmap'],kind='CII_box',fig=fig,ax=axs[0,j],
                                                zticks='redshift',aspect='equal',
                                                log=pspecs['CII_box']['lognorm'],slice_axis=2,slice_index=idx,
                                                vmin=pspecs['CII_box']['vmin'],vmax=pspecs['CII_box']['vmax'],)
            
            p21c.plotting.lightcone_sliceplot(lc,cmap=pspecs['brightness_temp_diff']['cmap'],kind='brightness_temp_diff',fig=fig,ax=axs[1,j],
                                                zticks='redshift',aspect='equal',
                                                log=pspecs['brightness_temp_diff']['lognorm'],slice_axis=2,slice_index=idx,
                                                vmin=pspecs['brightness_temp_diff']['vmin'],vmax=pspecs['brightness_temp_diff']['vmax'],)
        fix_ticks(fig)
        fig.tight_layout()
        fig.savefig(''.join(['.',] + outname.split('.')[:-1] + ['_diag.',] + [outname.split('.')[-1],]))

def plot_cii_xps(lcfile,outname,bt_thermal=False,bt_uvcov=False,bt_wedge='none',
                 cii_noisefile=None,n_psbins=24,z_max=50.,**kwargs
                 ):
    lc = p21c.LightCone.read(lcfile,safe=False)
    z_targets = np.array([6.5,9,11])
    bt_lc, cii_lc, k_arr, power_arr, z_ps, _ = cii_xps(lc,bt_thermal=bt_thermal,bt_uvcov=bt_uvcov,n_psbins=n_psbins,
                                            bt_wedge=bt_wedge,ps_z=z_targets,cii_noisefile=cii_noisefile,z_max=z_max)

    bt_title = '21cm Brightness Temperature'
    if bt_uvcov: bt_title += ' - Mean' 
    if bt_thermal: bt_title += ' + Noise'
    if bt_wedge != 'none': bt_title += ' - Wedge'

    make_cii_xps_plot(lc,z_ps,k_arr,power_arr,**kwargs)


def plot_cii(lcfile):
    lc = p21c.LightCone.read(lcfile,safe=False)
    cii_lc = make_cii_map(lc)
    
    #HACK to make zeros colored in
    cii_lc[cii_lc <= 0] = 1e-6

    #Replace fields
    setattr(lc,'CII_box',cii_lc)

    gs = gridspec.GridSpec(ncols=1,nrows=1)

    fig = plt.figure(figsize=(12,4))
    fig.subplots_adjust(top=0.99,bottom=0.1,left=0.1,right=0.95)
    ax = fig.add_subplot(gs[0,0])

    
    fig, ax = p21c.plotting.lightcone_sliceplot(lc,cmap='viridis',kind='CII_box',fig=fig,ax=ax,zticks='redshift',aspect='equal'
                                                ,log=True,cbar_label=r'CII (Jy sr$^{-1}$)',vertical=False,cbar_horizontal=True
                                                ,vmin=1e-3,vmax=1e3,z_range=[5,16],slice_index=24)
    
    ax.set_title('CII Surface Brightness Density')

    fig.savefig('./plots/cii_lc.png')



def plot_galx(lcfile):
    lc = p21c.LightCone.read(lcfile,safe=False)
    ps_z = np.array([6.5,8,10,12])
    z_lc = lc.lightcone_redshifts
    ps_idx = np.argmin(np.fabs(z_lc[None,:] - ps_z[:,None]),axis=1)
    bt_lc, gal_lc, ps_result = gal_xps(lc,bt_thermal=True,bt_uvcov=True,bt_wedge='None',ps_z=ps_z)

    bt_title = '21cm Brightness Temperature'
    bt_title += ' - Mean'
    bt_title += ' + Noise'
    #bt_title += ' - Wedge'

    galmin = gal_lc[gal_lc > 0].min()
    galmax = gal_lc.max()

    z_dim = bt_lc.shape[-1]
    x_dim = bt_lc.shape[0]

    gs = gridspec.GridSpec(ncols=2,nrows=2,width_ratios=(z_dim/x_dim/5,1),hspace=0.05,wspace=0.05)

    fig = plt.figure(figsize=(12,4))

    ax = fig.add_subplot(gs[0,0])
    im_cii = ax.imshow(gal_lc[0,...],cmap='viridis',norm=LogNorm(vmin=galmin,vmax=galmax))
    plt.colorbar(im_cii,ax=ax,label='')
    ax.set_title('CII Surface Brightness Density')
    [ax.axvline(p,color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline(p+lc.user_params.HII_DIM//2,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline(p-lc.user_params.HII_DIM//2,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]

    ax = fig.add_subplot(gs[1,0])
    im_bt = ax.imshow(bt_lc[0,...],cmap='RdBu',norm=Normalize(vmin=-30,vmax=30))
    plt.colorbar(im_bt,ax=ax,label='dTb (mK)')
    [ax.axvline(p,color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline(p+lc.user_params.HII_DIM//2,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    [ax.axvline(p-lc.user_params.HII_DIM//2,linewidth=0.5,linestyle='--',color=f'C{i:02d}') for i,p in enumerate(ps_idx)]
    ax.set_title(bt_title)

    ax = fig.add_subplot(gs[:,1])
    [ax.plot(k,ps*(k**3)/(2*np.pi),label=f'z={ps_z[i]}',color=f'C{i:02d}') for i,(ps,k) in enumerate(ps_result)]
    ax.set_yscale('symlog',linthresh=1e-2)
    ax.set_xscale('log')
    ax.set_xlabel('k (cMpc-1 h)')
    ax.set_ylabel(r'$\left< \delta_{21} \delta_{CII} \right> \frac{k^3}{2\pi}$')
    ax.set_title('Dimensionless 1-D Cross Power Spectrum')
    ax.grid()
    ax.legend()

    fig.subplots_adjust(top=0.95,bottom=0.1,left=0.05,right=0.95)

    fig.savefig('./plots/gal_xcorr_wedge.png')

def plot_xray_xps(lcfile):
    lc = p21c.LightCone.read(lcfile,safe=False)

    xps,k = xray_xps(lc)

    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(8,3))

    # axs[0].imshow(x_lc,cmap='viridis',norm=Normalize(vmin=0,vmax=3e-16),interpolation='nearest',extent=[0,200,0,200])
    # axs[1].imshow(bt_lc,cmap='plasma',norm=Normalize(),interpolation='nearest',extent=[0,200,0,200])
    axs[2].plot(k,xps,fmt='k-')

    fig.savefig('./plots/xray_xcorr.png')

def global_seriesplot(lc_list,kinds,zmax,names,output):
    lightcones = [p21c.LightCone.read(lcf,safe=False) for lcf in lc_list]
    register_eor_cmaps()
    
    #GLOBAL QUANTITY PLOTTING
    fig, axs = plt.subplots(len(kinds),1,figsize=(4,3*len(kinds)),layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0,
                            wspace=0.0)
    if len(kinds)==1:
        axs = [axs]

    lines =  ['-',':','--','-.','-',':','--','-.']
    for i,lc in enumerate(lightcones):
        if 'n_ion' in kinds:
            setattr(lc,'n_ion',getattr(lc,'n_ion')*(C.M_sun/C.m_p).to('').value)
        for j,g in enumerate(kinds):
            spec = pspecs[g]
            try:
                _, ax = p21c.plotting.plot_global_history(lc,kind=g,ylog=spec['lognorm'],color=f'C{i:d}',
                                                          linestyle=lines[i%len(lines)],ax=axs[j],label=names[i] if names else None,zmax=zmax)
                if g == 'xH_box':
                    ax.set_xlim(5,12)
            except AssertionError:
                pass
            
    if names:
        axs[0].legend(fontsize=6)
    fig.savefig(output)

def photoncons_plot(lc_list,output):
    lightcones = [p21c.LightCone.read(lcf,safe=False) for lcf in lc_list]
    #GLOBAL QUANTITY PLOTTING
    fig, axs = plt.subplots(2,1,figsize=(12,10))

    for i,lc in enumerate(lightcones):
        label = lc_list[i].split('lightcone_')[-1].split('.')[0]
        pcd = lc.photon_nonconservation_data
        # print(f'--------{label}--------')
        # print(pcd)
        # print('------------------------')

        try:
            ax = axs[0]
            _, ax = p21c.plotting.plot_global_history(lc,kind='xH_box',ylog=False,color=f'C{i:d}',ax=ax,label=label)
            ax.set_xlim(6,12)
            if "CONS" in label:
                ax.plot(pcd['z_analytic'],1-pcd['Q_analytic'],label='analtyic '+label,color=f'C{i:d}',linestyle='--')
                ax.plot(pcd['z_calibration'],pcd['nf_calibration'],label='calibration '+label,color=f'C{i:d}',linestyle=':')
            ax.legend(fontsize=6)

            ax = axs[1]
            if "CONS" in label:
                ax.plot(pcd['nf_photoncons'],pcd['delta_z_photon_cons'],label='pc '+label,color=f'C{i:d}',linestyle='-')
                z_interp_calib = np.interp(pcd['nf_photoncons'],pcd['nf_calibration'],pcd['z_calibration'])
                z_interp_analy = np.interp(pcd['nf_photoncons'],1-pcd['Q_analytic'],pcd['z_analytic'])
                print(z_interp_analy)
                print(z_interp_calib)
                ax.plot(pcd['nf_photoncons'],z_interp_analy-z_interp_calib,label='a-c '+label,color=f'C{i:d}',linestyle=':')
            ax.legend(fontsize=6)
        except AssertionError:
            pass
            
    fig.tight_layout()
    fig.savefig(output)

def lc_seriesplot(lc_list,kinds,zrange,output,names=None,vertical=True):
    #load lightcones
    lightcones = []
    for lc in lc_list:
        if isinstance(lc,str): 
            lc = p21c.LightCone.read(lc,safe=False)
        if not isinstance(lc,p21c.LightCone):
            raise ValueError("lc list must be either py21cmfast.LightCone or string")
        lightcones.append(lc)
    
    register_eor_cmaps()

    #find the z-axis (assume its the same for each lightcone)
    z_len = np.argmin(np.fabs(lightcones[0].lightcone_redshifts - zrange[1])) - \
            np.argmin(np.fabs(lightcones[0].lightcone_redshifts - zrange[0]))
    num_plots = len(lc_list) * len(kinds)
    nrows = 1 if vertical else num_plots
    ncols = 1 if not vertical else num_plots
    plot_dims = [8,8*(z_len+1)/getattr(lightcones[-1], kinds[0]).shape[1]/num_plots]
    plot_dims = plot_dims[::-1] if not vertical else plot_dims
                            
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=plot_dims,
                            layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=2 / 72, hspace=0.0,
                            wspace=0.0)

    if num_plots == 1:
        axs = [axs,]

    cb_indiv = len(kinds) > 1
    for i,lc in enumerate(lightcones):
        if 'n_ion' in kinds:
            rhocrit = (lc.cosmo_params.cosmo.critical_density(0)).to('M_sun Mpc-3').value
            nion_converted = getattr(lc,"n_ion") / rhocrit / lc.cosmo_params.OMb
            setattr(lc,'n_ion',nion_converted)
        for j,kind in enumerate(kinds):
            xyz = getattr(lc,kind)
            logger.info(f" LC {i}: {kind}: MIN {xyz.min():.3e}, MAX {xyz.max():.3e}, MEAN {xyz.mean():.3e}")
            spec = find_pspec(kind,lc)
            ax_idx = j+i*len(kinds)
            fig, ax = p21c.plotting.lightcone_sliceplot(lc,cmap=spec['cmap'],kind=kind,fig=fig,ax=axs[ax_idx],
                                                        zticks='redshift',aspect='equal',cbar=cb_indiv,
                                                        log=spec['lognorm'],cbar_label=spec['clabel'],vertical=vertical,
                                                        cbar_horizontal=vertical,vmin=spec['vmin'],vmax=spec['vmax'],z_range=zrange)
            if names:
                axs[ax_idx].text(0.05,0.99,names[i],transform=axs[ax_idx].transAxes,verticalalignment='top',horizontalalignment='left',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    fix_ticks(fig,sharex=True,sharey=True)

    if not cb_indiv:
        cb = plt.colorbar(ax.get_images()[0],ax=fig.get_axes(),aspect=10*num_plots,
                     fraction=0.01*num_plots,pad=0.05,orientation='horizontal')
        cb.ax.set_xlabel(spec['clabel'],fontsize=8)

    fig.savefig(output)

def coev_seriesplot(coev_list,kinds,output,slice_idx=0):
    #load lightcones
    register_eor_cmaps()
    n_cv = len(coev_list)
    n_kind = len(kinds)
    
    #make the plot
    fig, axs = plt.subplots(n_kind, n_cv,
                            figsize=(12,9*n_kind/n_cv),squeeze=False,
                            layout='constrained')
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0,
                            wspace=0.0)
    
    cb_indiv = len(kinds) > 1
    for i,cvf in enumerate(coev_list):
        cv = p21c.Coeval.read(cvf,safe=False)
        logger.info(f'starting {cvf}')
        axs[0,i].set_title(coev_list[i].split('coeval_')[-1].split('.')[0],fontsize=8)
        for j,kind in enumerate(kinds):
            spec = pspecs[kind]
            fig, ax = p21c.plotting.coeval_sliceplot(cv,cmap=spec['cmap'],kind=kind,fig=fig,ax=axs[j,i],aspect='equal',cbar=cb_indiv
                                                        ,log=spec['lognorm'],cbar_label=spec['clabel'],vmin=spec['vmin'],vmax=spec['vmax'],slice_index=slice_idx)

    if not cb_indiv:
        cb = plt.colorbar(ax.get_images()[0],ax=fig.get_axes(),aspect=10*num_plots,
                     fraction=0.01*num_plots,pad=0.05,orientation='horizontal')
        cb.ax.set_xlabel(spec['clabel'],fontsize=8)

    logger.info(f'saving {output}')
    fig.savefig(output)

def largescale_powerplot(lc_list,output='',names=None,k_target=(0.1,),z_max=16,kind='brightness_temp',kind2=None,
                        n_psbins=20,ps_cadence=100,subplot_k=True,k_lab=None,axs=None):    
    #we use the first lightcone to get the redshift targets
    #  this doesn't assume that they are the same size but cadence may vary
    #  if there are lightcones of very differenct sizes in the list
    n_lc = len(lc_list)
    lc =  lc_list[0] if isinstance(lc_list[0],p21c.LightCone) else p21c.LightCone.read(lc_list[0],safe=False)
    lc_z = lc.lightcone_redshifts
    hii_dim = lc.user_params.HII_DIM
    dim_buffer = int(hii_dim/2)
    z_targets = lc_z[dim_buffer:-dim_buffer:ps_cadence]
    z_targets = z_targets[z_targets < z_max]

    if names is None:
        names = ['',]*n_lc

    k_arr, power_arr, z_ps = get_lc_powerspectra(lc_list,z_targets,kind,kind2=kind2,n_psbins=n_psbins)

    #we make sure here that we found the same k at each redshift
    #we allow different k for different lightcones to allow different hii_dim
    k_check = (k_arr == k_arr[:,:1,:]) #(lc,z,k)
    if not np.all(k_check):
        logger.error(f"different k at {np.where(~k_check)}")
        logger.error(f"indexded: {k_arr[~k_check]}")
        raise ValueError(f"We found different k at different redshifts.")

    k_arr = k_arr[:,0,:]
    k_target = np.array(k_target)
    
    k_idx = np.argmin(np.fabs(k_arr[:,:,None] - k_target[None,None,:]),axis=-2) #(lc,(k/k_target)) --> (lc,k_target)
    lc_idx = np.mgrid[0:n_lc,0:k_target.size]
    full_idx = np.concatenate((lc_idx,k_idx[None,...]),axis=0) #(axis,lc,target)
    k_plot = k_arr[full_idx[0],full_idx[2]]

    ax_mode = True
    n_plots = k_target.size if subplot_k else len(lc_list)
    if axs is None:
        ax_mode = False
        fig,axs = plt.subplots(1,n_plots,figsize=(8,8/n_plots),sharey=True,layout='constrained')
        fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0,
                                wspace=0.0)
    
    k_text = []
    if k_lab is None:
        k_text = [rf'$k={{{kt:.2f}}}$' for kt in k_target]
    else:
        k_text = k_lab

    for i,ax in enumerate(axs):
        ax.set_ylabel(r'$\Delta_{21} \overline{dT}_{b,21} (mK^2)$')
        ax.set_xlabel(f'z')
        ax.set_xlim(z_ps.min(),z_ps.max())
        ax.set_ylim(power_arr[:,:,k_idx[0,i]].max()/100,power_arr[:,:,k_idx[0,i]].max())
        ax.grid()
        ax_text = k_text[i] if subplot_k else names[i]
        axs[i].text(0.05,0.95,ax_text,transform=axs[i].transAxes,verticalalignment='top',horizontalalignment='left',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    lines =  ['-',':','--','-.','-',':','--','-.']
    for i,k_t in enumerate(k_target):
        for j in range(n_lc):
            ax_idx = i if subplot_k else j
            line_idx = j if subplot_k else i
            axs[ax_idx].semilogy(z_ps[j,:],power_arr[j,:,k_idx[j,i]],color=f'C{line_idx:d}',linestyle=lines[line_idx],
                                 label=names[j] if subplot_k else k_text[i],
                                 linewidth=1.5)
        
    axs[0].legend(loc=4)
    if not ax_mode:
        fix_ticks(fig)
        fig.savefig(output)


def powerspec_plot(lc_list,output='',names=None,z_out=(11,9,7),kind='brightness_temp',kind2=None,
                   n_psbins=20,z_type='redshift',subplot_z=True,z_lab=None,axs=None):
    z_out = np.array(z_out)
    if isinstance(z_type,str):
        z_type = [z_type,]*z_out.size

    k_arr, power_arr, z_ps = get_lc_powerspectra(lc_list,z_out,kind,kind2=kind2,n_psbins=n_psbins,z_type=z_type)

    n_plots = z_ps.shape[1] if subplot_z else len(lc_list)
    save_fig = False
    if axs is None:
        save_fig = True
        fig,axs = plt.subplots(1,n_plots,figsize=(8,8/n_plots),gridspec_kw={'hspace':0.},sharey=True,sharex=True,layout='constrained',squeeze=False)
        axs=axs.flatten()
        fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0,
                                wspace=0.0)
    
    single_lc = len(lc_list) == 1

    #setup default labels
    if z_lab is None:
        z_lab = []
        for z,zt in zip(z_out,z_type):
            if zt.lower() == 'redshift':
                plottxt = rf'$z \sim~$ {z}'
            elif zt.lower() == 'xhi':
                plottxt = rf'$X_{{HI}} \sim$ {z}'
            elif zt.lower() == 'bt_min':
                plottxt = rf'$\delta T_b$ Min.'
            elif zt.lower() == 'bt_max':
                plottxt = rf'$\delta T_b$ Max.'
            z_lab.append(plottxt)

    if names is None:
        names = []
        for lc in lc_list:
            if isinstance(lc,str):
                names.append(lc)
            else:
                names.append('')

    #setup the axes
    for i,ax in enumerate(axs):
        ax.set_xlim(k_arr[:,:,1:].min(),k_arr.max())
        ax.set_ylabel(r'$\Delta_{21} \overline{dT}_{b,21} (mK^2)$')
        ax.set_xlabel(r'k (Mpc-1)')
        ax.grid()
            
        plottxt = names[i] if not subplot_z else z_lab[i]
            
        axs[i].text(0.05,0.95,plottxt,transform=axs[i].transAxes,verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
        axs[i].set_xscale('log')
        #crosses can go negative
        if kind2 is not None and kind2 != kind:
            axs[i].set_yscale('symlog',linthresh=2e-1)
        else:
            axs[i].set_yscale('log')

    lines =  ['-',':','--','-.','-',':','--','-.']
    for i,(z,zt) in enumerate(zip(z_out,z_type)):
        for j in range(len(lc_list)):
            ax_idx = i if subplot_z else j
            line_idx = j if subplot_z else i
            axs[ax_idx].plot(k_arr[j,i,:],power_arr[j,i,:],color=f'C{line_idx:d}',linestyle=lines[line_idx],
                                label=names[j] if subplot_z else z_lab[i],linewidth=1.5)

    if names[0]:
        axs[0].legend(loc=4)
    if save_fig:
        fix_ticks(fig,sharex=True,sharey=True)
        fig.savefig(output)


def scaling_plot(params,redshifts,propx,propy,seed=None):
    nrows = len(redshifts)
    ncols = len(propx)
    if not len(propy) == ncols:
        logger.error(f"property arrays must be same length, array lengths: {len(propx)}, {len(propy)}")
        return 0

    if seed is None:
        seed = np.random.randint(100000,size=1)

    # df_serra,q_serra = read_serra('data/serra_presentation_paper_data.csv')
    # df_astrid,q_astrid = read_astrid(6)
    # df_astrid12,q_astrid12 = read_astrid(12)

    fig,axs = plt.subplots(figsize=(8,6),nrows=nrows,ncols=ncols,squeeze=False)

    for i,(cp,up,ap,fo) in enumerate(params):
        init_box = p21c.initial_conditions(
            user_params=up,
            cosmo_params=cp,
            random_seed=seed,
        )
        for j,z in enumerate(redshifts):
            halolist = p21c.determine_halo_list(
                redshift=z,
                halos_desc=halos_desc,
                init_boxes=init_box,
                astro_params=ap,
                flag_options=fo,
            )
            halos_desc = halolist
            for k,(px,py) in enumerate(zip(propx,propy)):
                arr_x = getattr(halolist,px)
                arr_y = getattr(halolist,py)

                axs[j,k].scatter(arr_x,arr_y)

def angular_lc_plot(lc,lcn,fname,kinds=('brightness_temp',)):
    lath = lcn.latitude
    lonh = lcn.longitude
    plt.style.use('dark_background')
    register_eor_cmaps()
    
    H, D = np.meshgrid(lonh, lcn.lc_distances.value,indexing='ij')

    fig, ax = plt.subplots(1,1,figsize=[12, 10], constrained_layout=True,subplot_kw={'projection': 'polar'})
    ax.set_thetamax(lonh.max() * 180/np.pi)
    ax.set_ylim(lcn.lc_distances.min().value, lcn.lc_distances.max().value)
    ax.set_rorigin(0)

    print(f'LC SHAPE {lc.brightness_temp.shape} H {H.shape} D {D.shape}')
    for j,kind in enumerate(kinds):
        i_min = int(j/(len(kinds))*lonh.size)
        i_max = int((j+1)/(len(kinds))*lonh.size)
        print(f'Kind {kind} | indices [{i_min},{i_max}]')
        vmin, vmax = pspecs[kind]['vmin'], pspecs[kind]['vmax']
        
        img = ax.pcolormesh(H[i_min:i_max,:], D[i_min:i_max,:],
                                getattr(lc,kind)[i_min:i_max,:],
                                edgecolors='face', cmap=pspecs[kind]['cmap'],
                                shading='auto', vmin=vmin, vmax=vmax)
        
    ax.set_thetagrids(np.arange(len(kinds))/len(kinds) * lonh.max() * 180/np.pi)
    rgrid = np.arange(0, D.max(), lc.user_params.BOX_LEN)
    rgrid = rgrid[rgrid > D.min()]
    rgrid = np.append(rgrid, D.max())
    ax.set_rgrids(rgrid, labels=[f"{z_at_value(lcn.cosmo.comoving_distance, d*U.Mpc).value:.1f}" for d in rgrid])
    ax.grid(True)
    #ax.set_xlabel("Redshift")
    ax.text(0.5, -0.06, "Redshift", fontsize=14, transform=ax.transAxes)

    fig.savefig(fname)

