import py21cmfast as p21c

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import numpy as np

def make_test_sourceboxes(*,
    redshifts=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    direc=None,
    init_boxes=None,
    random_seed=None,
    halomass = 1e11,
    halo_redshift=None
):
    p21c.config['direc'] = direc
    boxes = []
    redshifts = np.sort(redshifts)[::-1]
    halo_zidx = np.argmin(np.fabs(redshifts-halo_redshift)) if halo_redshift is not None else None
    midpoint = int(user_params.HII_DIM // 2)
    for i,z in enumerate(redshifts):
        #init the class
        box = p21c.outputs.HaloBox(
            redshift=z,
            random_seed=random_seed,
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
        )

        #call to allocate memory
        box()
        
        #put one halo in the middle of the box
        
        if i == halo_zidx or halo_redshift is None:
            star = halomass * cosmo_params.OMb / cosmo_params.OMm * max(astro_params.F_STAR10 * (halomass/1e10)**(astro_params.ALPHA_STAR),1)
            sfr = star / astro_params.t_STAR * cosmo_params.cosmo.H(z).to('s-1').value
            sfr_mini = sfr / 1e3 #hacky

            box.halo_mass[midpoint,midpoint,midpoint] = halomass
            box.halo_sfr[midpoint,midpoint,midpoint] = sfr
            box.halo_sfr_mini[midpoint,midpoint,midpoint] = sfr_mini
        
        box.log10_Mcrit_LW_ave = 1e7 #needed for calculation but unused
            #not using the rest of the fields yet
        
        box() #call again to pass in

        for k, state in box._array_state.items():
            if state.initialized:
                state.computed_in_mem = True

        boxes.append(box)


    xray_sourcebox = p21c.xray_source(
                                    redshift=redshifts[-1],
                                    cosmo_params=cosmo_params,
                                    user_params=user_params,
                                    astro_params=astro_params,
                                    flag_options=flag_options,
                                    hboxes=boxes,
                                    init_boxes=init_boxes,
                                    z_halos=redshifts,
                                    write=False,
                                    direc='./')

    return xray_sourcebox

def xray_source_plot(box,R_plot_idx=None,R_plot_value=None,fname='xrs_plot.png'):
    R_min = box.user_params.BOX_LEN/box.user_params.HII_DIM * 0.620350491
    R_steps = np.arange(0, p21c.global_params.NUM_FILTER_STEPS_FOR_Ts)
    R_factor = (p21c.global_params.R_XLy_MAX / R_min) ** (
        R_steps / p21c.global_params.NUM_FILTER_STEPS_FOR_Ts
    )
    R_box = R_min * R_factor 
    if R_plot_idx is None:
        if R_plot_value is None:
            R_plot_idx = np.arange(box.filtered_sfr.shape[0])
        else:
            R_plot_idx = np.searchsorted(R_box,R_plot_value)

    R_plot_value = R_box[R_plot_idx]
    #try to make a nice aspect plot
    nrows = 1
    ncols = 1
    for ii in range(1,len(R_plot_idx)+1):
        ratio_test = ncols/nrows
        if ii > nrows * ncols:
            if ratio_test < 4/3.:
                ncols += 1
            else:
                nrows += 1

    fig,axs = plt.subplots(nrows,ncols,figsize=(8,6))
    axs = axs.flatten()

    midpoint = int(box.user_params.HII_DIM // 2)
    boxlen = box.user_params.BOX_LEN
    for i in range(len(R_plot_idx)):
        im = axs[i].imshow(box.filtered_sfr[R_plot_idx[i],midpoint,:,:],origin='lower',extent=[0,boxlen,0,boxlen],norm=matplotlib.colors.LogNorm(vmin=1e-11,vmax=1e-5))
        axs[i].set_title(f'R  = {R_plot_value[i]:.2f}')

    fig.tight_layout()
    plt.colorbar(im,ax=axs)
    fig.savefig(fname)
