#downsample a 3d array by a factor
import numpy as np
from scipy.stats import binned_statistic as binstat
from powerbox.tools import get_power
from copy import deepcopy

import py21cmfast as p21c
from py21cmfast.c_21cmfast import ffi,lib
from astropy import units as U

from scipy.ndimage import gaussian_filter1d, uniform_filter1d

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def lightcone_axis_mean(lc,kind='brightness_temp',smooth_cells=3,smooth_filter='boxcar'):
    lc_z_means = getattr(lc,kind).mean(axis=(0,1))
    if smooth_filter == 'boxcar':
        lc_z_means = uniform_filter1d(lc_z_means,smooth_cells,mode='reflect')
    if smooth_filter == 'gaussian':
        lc_z_means = gaussian_filter1d(lc_z_means,smooth_cells,mode='reflect')
    return lc_z_means

#Generates fake ionized, spintemp and perturbed halo fields for getting fields w/o feedback
def get_fake_boxes(inputs,redshift):
    #fake ion field
    fake_ion = p21c.IonizedBox(
        redshift=redshift,
        inputs=inputs,
    )
    fake_ion() #allocate memory
    fake_ion.z_re_box[...] = -1.
    fake_ion.Gamma12_box[...] = 0.
    fake_ion() #pass to C

    for k, state in fake_ion._array_state.items():
            if state.initialized:
                    state.computed_in_mem = True

    #fake ts field
    fake_ts = p21c.TsBox(
        redshift=redshift,
        inputs=inputs,
    )
    fake_ts() #allocate memory
    fake_ts.J_21_LW_box[...] = 0.
    fake_ts() #pass to C
    
    for k, state in fake_ts._array_state.items():
        if state.initialized:
            state.computed_in_mem = True
    
    #fake pt halos for subsampler box
    fake_pth = p21c.PerturbHaloField(
        redshift=redshift,
        inputs=inputs,
        buffer_size=0,
    )
    fake_pth() #allocate memory
    fake_pth.n_halos = int(0)
    fake_pth() #pass to C

    for k, state in fake_pth._array_state.items():
            if state.initialized:
                    state.computed_in_mem = True

    return fake_pth,fake_ts,fake_ion

#generates HaloField, PerturbHaloField, and HaloBox at a desired redshift
#  without reionisation / feedback
def get_halo_fields(redshift,inputs,init_box,ptb,lagrangian=False):
    halos_desc = None
    halo_field = None
    pt_halos = None
    z_array = np.append(inputs.node_redshifts[inputs.node_redshifts < redshift],redshift)
    logger.info(f'getting halos at {z_array}')
    if not inputs.flag_options.FIXED_HALO_GRIDS:
        for z in z_array:
            halo_field = p21c.determine_halo_list(
                redshift=z,
                inputs=inputs,
                initial_conditions=init_box,
                regenerate=True,
                halos_desc=halos_desc,
                write=False,
            )
            halos_desc = halo_field
        
        n_halos = halo_field.n_halos
        if lagrangian:
            dim_fac = halo_field.user_params.HII_DIM / halo_field.user_params.DIM
            pt_halos = p21c.PerturbHaloField(
                redshift=z,
                inputs=inputs,
                buffer_size=n_halos,
            )
            pt_halos()
            pt_halos.n_halos = n_halos
            setattr(pt_halos.cstruct,'n_halos',n_halos)
            pt_halos.halo_masses[...] = halo_field.halo_masses[:n_halos]
            pt_halos.halo_coords[...] = (halo_field.halo_coords[:n_halos,:]*dim_fac).astype('i4')
            pt_halos.star_rng[...] = halo_field.star_rng[:n_halos]
            pt_halos.sfr_rng[...] = halo_field.sfr_rng[:n_halos]
            pt_halos.xray_rng[...] = halo_field.xray_rng[:n_halos]
            # HACK: Since we don't compute, we have to mark the struct as computed
            for k, state in pt_halos._array_state.items():
                if state.initialized:
                    state.computed_in_mem = True
        else:
            pt_halos = p21c.perturb_halo_list(
                redshift=redshift,
                inputs=inputs,
                initial_conditions=init_box,
                halo_field=halo_field,
                regenerate=True,
                write=False
            )

    fpth,fts,fion = get_fake_boxes(redshift=redshift,inputs=inputs)
    hbox = p21c.compute_halo_grid(
        initial_conditions=init_box,
        inputs=inputs,
        perturbed_halo_list=pt_halos,
        perturbed_field=ptb,
        previous_spin_temp=fts,
        previous_ionize_box=fion,
        write=False,
        regenerate=True,
    )
    hbox_ss = p21c.compute_halo_grid(
        inputs=inputs,
        initial_conditions=init_box,
        perturbed_halo_list=fpth,
        perturbed_field=ptb,
        previous_spin_temp=fts,
        previous_ionize_box=fion,
        write=False,
        regenerate=True,
    )
    return halo_field,pt_halos,hbox,hbox_ss

#gets properties from either HaloField or PerturbHaloField
#ignoring feedback and with possible alterations to parameters
def get_props_from_halofield(halo_field,inputs,sel=None,kinds=['sfr',]):
    #fake pt_halos
    zero_array = ffi.cast("float *", np.zeros(halo_field.user_params.HII_DIM**3,dtype='f4').ctypes.data)

    if sel is None:
        sel = slice(0,halo_field.n_halos)
    n_halos = halo_field.halo_masses[sel].size

    pt_halos = p21c.PerturbHaloField(
        inputs=inputs,
        buffer_size=n_halos
    )
    pt_halos()
    pt_halos.n_halos = n_halos
    setattr(pt_halos.cstruct,'n_halos',n_halos)
    pt_halos.halo_masses[...] = halo_field.halo_masses[sel]
    pt_halos.star_rng[...] = halo_field.star_rng[sel]
    pt_halos.sfr_rng[...] = halo_field.sfr_rng[sel]
    pt_halos.xray_rng[...] = halo_field.xray_rng[sel]

    #get props w/o feedback
    props_out = np.zeros(int(12*pt_halos.n_halos)).astype('f4')
    lib.test_halo_props(
        halo_field.redshift,
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct,
        zero_array,
        zero_array,
        zero_array,
        zero_array,
        pt_halos(),
        ffi.cast("float *", props_out.ctypes.data),
    )

    props_out = props_out.reshape((pt_halos.n_halos,12))
    
    sel = np.any(~np.isfinite(props_out),axis=-1)
    if sel.sum() > 0:
            print(f'{sel.sum()} invalid halos')
            print(f'First 10: {props_out[sel,:][:10,:]}')

    index_map = {'mass' : 0, 'star' : 1, 'sfr' : 2, 'xray' : 3, 'nion' : 4, 'wsfr' : 5,
                 'star_mini' : 6, 'sfr_mini' : 7, 'mturn_a' : 8, 
                 'mturn_m' : 9, 'mturn_r' : 10, 'metallicity' : 11}
    
    return {kind:props_out[:,index_map[kind]] for kind in kinds}

def get_cell_integrals(redshift,inputs,delta,lnMmin,lnMmax):
    #pass info to backend
    lib.Broadcast_struct_global_all(
        inputs.user_params.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.flag_options.cstruct
    )
    lib.init_ps()

    ap_c = inputs.astro_params.cdict

    rhocrit = inputs.cosmo_params.cosmo.critical_density(0)
    growth = lib.dicke(redshift)
    lnMmin = np.log(inputs.user_params.SAMPLER_MIN_MASS)
    lnMmax = np.log(1e15) #must be > cell but < table max
    cell_volume = (inputs.user_params.BOX_LEN * U.Mpc / inputs.user_params.HII_DIM) ** 3
    cell_mass = (rhocrit * inputs.cosmo_params.OMm * cell_volume).to("M_sun")
    sigma_cell = lib.sigma_z0(cell_mass.value)
    Mlim_Fstar = 1e10 * (ap_c['F_STAR10']) ** (-1.0 / ap_c['ALPHA_STAR'])
    Mlim_Fesc = 1e10 * (ap_c['F_ESC10']) ** (-1.0 / ap_c['ALPHA_ESC'])
    
    Mlim_Fstar_MINI = 1e7 * (ap_c['F_STAR7_MINI']) ** (-1.0 / ap_c['ALPHA_STAR_MINI'])
    Mlim_Fesc_MINI = 1e7 * (ap_c['F_ESC7_MINI']) ** (-1.0 / ap_c['ALPHA_ESC'])

    Mturn_acg = lib.atomic_cooling_threshold(redshift) if inputs.flag_options.USE_MINI_HALOS else ap_c['M_TURN']
    Mturn_mcg = ap_c['M_TURN'] # no LW

    t_h = 1 / inputs.cosmo_params.cosmo.H(redshift).to('s-1').value

    if lnMmax < 0:
        lnMMax = cell_mass
        
    if (
        inputs.user_params.INTEGRATION_METHOD_ATOMIC == 'GAUSS-LEGENDRE'
        or inputs.user_params.INTEGRATION_METHOD_MINI == 'GAUSS-LEGENDRE'
    ):
        lib.initialise_GL(lnMmin, lnMmax)

    #NOTE: Minihalo integrals assume no feedback
    m_integral = lib.Mcoll_Conditional(
            growth,
            lnMmin,
            lnMmax,
            np.log(cell_mass.value),
            sigma_cell,
            delta,
            0,
    )
    integral_stars = lib.Nion_ConditionalM(
        growth,
        lnMmin,
        lnMmax,
        np.log(cell_mass.value),
        sigma_cell,
        delta,
        Mturn_acg,
        ap_c['ALPHA_STAR'],
        ap_c['ALPHA_ESC'],
        ap_c['F_STAR10'],
        ap_c['F_ESC10'],
        Mlim_Fstar,
        Mlim_Fesc,
        inputs.user_params.cdict['INTEGRATION_METHOD_ATOMIC'],
    )

    integral_fesc = lib.Nion_ConditionalM(
        growth,
        lnMmin,
        lnMmax,
        np.log(cell_mass.value),
        sigma_cell,
        delta,
        Mturn_acg,
        ap_c['ALPHA_STAR'],
        ap_c['ALPHA_ESC'],
        ap_c['F_STAR10'],
        ap_c['F_ESC10'],
        Mlim_Fstar,
        Mlim_Fesc,
        inputs.user_params.cdict['INTEGRATION_METHOD_ATOMIC'],
    )
    
    if inputs.flag_options.USE_MINI_HALOS:
        integral_stars_MINI = lib.Nion_ConditionalM_MINI(
            growth,
            lnMmin,
            lnMmax,
            np.log(cell_mass.value),
            sigma_cell,
            delta,
            Mturn_mcg,
            Mturn_acg,
            ap_c['ALPHA_STAR_MINI'],
            0.,
            ap_c['F_STAR7_MINI'],
            1.,
            Mlim_Fstar,
            0.,
            inputs.user_params.cdict['INTEGRATION_METHOD_ATOMIC'],
        )

        integral_fesc_MINI = lib.Nion_ConditionalM_MINI(
            growth,
            lnMmin,
            lnMmax,
            np.log(cell_mass.value),
            sigma_cell,
            delta,
            Mturn_mcg,
            Mturn_acg,
            ap_c['ALPHA_STAR_MINI'],
            ap_c['ALPHA_ESC'],
            ap_c['F_STAR7_MINI'],
            ap_c['F_ESC7_MINI'],
            Mlim_Fstar,
            Mlim_Fesc,
            inputs.user_params.cdict['INTEGRATION_METHOD_MINI'],
        )
        
    integral_xray = lib.Xray_ConditionalM(
        redshift,
        growth,
        lnMmin,
        lnMmax,
        np.log(cell_mass.value),
        sigma_cell,
        delta,
        Mturn_mcg,
        Mturn_acg,
        ap_c['ALPHA_STAR'],
        ap_c['ALPHA_STAR_MINI'],
        ap_c['F_STAR10'],
        ap_c['F_STAR7_MINI'],
        Mlim_Fstar,
        Mlim_Fstar_MINI,
        ap_c['L_X'],
        ap_c['L_X_MINI'],
        t_h,
        ap_c['t_STAR'],
        inputs.user_params.cdict['INTEGRATION_METHOD_ATOMIC'],
    )

    out = {
        'mass' : m_integral,
        'stars' : integral_stars,
        'stars_mini' : integral_stars_MINI if inputs.flag_options.USE_MINI_HALOS else None,
        'fesc' : integral_fesc,
        'fesc_mini' : integral_fesc_MINI if inputs.flag_options.USE_MINI_HALOS else None,
        'xray' : integral_xray,
    }
    return out


#gets expected HaloBox Values given a cell delta.
#  Should be equivalent to calling backend (set_fixed_grids)
#  on one cell w/o feedback
def get_expected_lagrangian(redshift,inputs,delta,lnMmin,lnMmax):
    lib.initialiseSigmaMInterpTable(inputs.user_params.SAMPLER_MIN_MASS, 1e16)

    ap_c = inputs.astro_params.cdict
    #setup cell values & constants
    t_h = (1 / inputs.cosmo_params.cosmo.H(redshift)).to('s').value
    s_per_yr = 60 * 60 * 24 * 365.25

    rhocrit = inputs.cosmo_params.cosmo.critical_density(0).to('Msun Mpc-3').value
    prefactor_mass = rhocrit * inputs.cosmo_params.OMm
    prefactor_stars = rhocrit * inputs.cosmo_params.OMb * ap_c['F_STAR10']
    prefactor_sfr = prefactor_stars / ap_c['t_STAR'] / t_h
    prefactor_nion = prefactor_stars * ap_c['F_ESC10'] * p21c.global_params.Pop2_ion
    prefactor_wsfr = prefactor_sfr * ap_c['F_ESC10']
    prefactor_xray = rhocrit * inputs.cosmo_params.OMm

    #TODO: add minihalos
    # prefactor_stars_mini = rhocrit * cosmo_params.OMb * astro_params.F_STAR7
    # prefactor_sfr_mini = prefactor_stars_mini / astro_params.t_STAR / t_h
    # prefactor_nion_mini = prefactor_stars_mini * astro_params.fesc_7 * p21c.global_params.Pop3_ion
    # prefactor_wsfr_mini = prefactor_sfr_mini * astro_params.fesc_7
    # prefactor_xray_mini = prefactor_sfr_mini * astro_params.l_x_mini * s_per_yr

    cint_dict = get_cell_integrals(redshift,inputs,delta,lnMmin,lnMmax)

    results = {
        'halo_mass': cint_dict['mass'] * prefactor_mass,
        'halo_stars' : cint_dict['stars'] * prefactor_stars,
        'halo_sfr' : cint_dict['stars'] * prefactor_sfr,
        'n_ion' : cint_dict['fesc'] * prefactor_nion,
        'fescweighted_sfr' : cint_dict['fesc'] * prefactor_wsfr,
        'halo_xray' : cint_dict['xray'] * prefactor_xray,
    }

    return results

def get_expected_eulerian(redshift,user_params,cosmo_params,astro_params,flag_options,delta,convert=False):
    #Either we apply the Mo & White Conversion to get the Lagrangian back,
    # OR we do what 21cmfastv3 does and pretend we're Lagrangian then multiply by (1+delta)
    if convert:
        dp1 = delta + 1
        delta_0 = -1.35*dp1**-2./3. + 0.78785*dp1**-0.58661 - 1.12431*dp1**-0.5 + 1.68647
    else:
        delta_0 = delta

    results = get_expected_lagrangian(redshift,user_params,cosmo_params,astro_params,flag_options,delta_0)
    if not convert:
        results.update((x,y*(1+delta)) for x,y in results.items())
    return results

#makes a grid from a halo catalogue by direct summation
def grid_halo_cat(halo_cat,inputs):
    lores = halo_cat.user_params.HII_DIM
    hires = halo_cat.user_params.DIM
    cell_volume = (
        halo_cat.user_params.BOX_LEN
         * U.Mpc
          / halo_cat.user_params.HII_DIM
    ) ** 3
    bins_crd = np.linspace(-0.5,lores+0.5,num=lores+1)
    masses = getattr(halo_cat,'halo_masses')[0:halo_cat.n_halos+1]
    crd = getattr(halo_cat,'halo_coords')[0:halo_cat.n_halos+1]
    crd = crd.reshape(masses.size,3)
    crd = crd / hires * lores

    props =  get_props_from_halofield(halo_cat,inputs,kinds=['nion','xray'])
    logger.info(f'crd {crd.shape} n {props["nion"].shape}')
    mass_grid,_ = np.histogramdd(crd,bins=[bins_crd,]*3,weights=masses)
    nion_grid,_ = np.histogramdd(crd,bins=[bins_crd,]*3,weights=props['nion'])
    xray_grid,_ = np.histogramdd(crd,bins=[bins_crd,]*3,weights=props['xray'])
    mass_grid = mass_grid * U.M_sun / cell_volume
    nion_grid = nion_grid / cell_volume
    xray_grid = xray_grid  * U.erg / U.s / cell_volume * 1e38

    logger.info('done')

    return mass_grid,nion_grid,xray_grid

def downsample(arr,dims):
    if len(arr.shape) != len(dims):
        logger.error('dimensions don\'t match')
        raise ValueError

    #TODO: a more numpythonic approach to finding reshape dims
    rdims = np.array([],dtype=int)
    for i,d in enumerate(dims):
        rdims = np.append(rdims,d) #add target dimension
        rdims = np.append(rdims,arr.shape[i]//d) #add arr/target dimension

    #rdims = (target_1,remainder_1,...)

    #TODO: generalise for non-3d
    sumaxes = (1,3,5)
    out = arr.reshape(rdims).mean(axis=sumaxes)
    out = out.reshape(dims)
    return out
    
#return binned mean & 1-2 sigma quantiles
def get_binned_stats(x_arr,y_arr,bins,stats):
    x_in = x_arr.flatten()
    y_in = y_arr.flatten()
    result = {}

    statistic_dict = {
        'pc1u' : lambda x: np.percentile(x,84),
        'pc1l' : lambda x: np.percentile(x,16),
        'pc2u' : lambda x: np.percentile(x,97.5),
        'pc2l' : lambda x: np.percentile(x,2.5),
        'err1u' : lambda x: np.percentile(x,84) - np.mean(x),
        'err2u' : lambda x: np.percentile(x,97.5) - np.mean(x),
        'err1l' : lambda x: np.mean(x) - np.percentile(x,16),
        'err2l' : lambda x: np.mean(x) - np.percentile(x,2.5),
    }

    for stat in stats:
        spstatkey = statistic_dict[stat] if stat in statistic_dict.keys() else stat
        result[stat],_,_ = binstat(x_in,y_in,bins=bins,statistic=spstatkey)

    return result

#Lightcone power spectra taken from 21CMMC for testing
def compute_power(
        box,
        length,
        box2=None,
        n_psbins=None,
        n_dim=1,
        log_bins=True,
        ignore_kperp_zero=True,
        ignore_kpar_zero=False,
        ignore_k_zero=False,
        remove_shotnoise=False,
    ):
        """Compute power spectrum from coeval box.
        Parameters
        ----------
        box : box to calculate power spectrum on
        length : 3-tuple
            Size of the lightcone in its 3 dimensions (X,Y,Z)
        n_psbins : int
            Number of power spectrum bins to return.
        log_bins : bool, optional
            Whether the bins are regular in log-space.
        ignore_kperp_zero : bool, optional
            Whether to ignore perpendicular k=0 modes when performing spherical average.
        ignore_kpar_zero : bool, optional
            Whether to ignore parallel k=0 modes when performing spherical average.
        ignore_k_zero : bool, optional
            Whether to ignore the ``|k|=0`` mode when performing spherical average.
        Returns
        -------
        power : ndarray
            The power spectrum as a function of k
        k : ndarray
            The centres of the k-bins defining the power spectrum.
        """
        # Determine the weighting function required from ignoring k's.
        if box2 is None:
            box2 = box

        k_weights = np.ones(box.shape, dtype=int)
        n0 = k_weights.shape[0]
        n1 = k_weights.shape[-1]

        if ignore_kperp_zero:
            k_weights[n0 // 2, n0 // 2, :] = 0
        if ignore_kpar_zero:
            k_weights[:, :, n1 // 2] = 0
        if ignore_k_zero:
            k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

        #number of dimensions to average over
        n_avgdim = len(box.shape) - n_dim + 1

        p_k,k = get_power(
            box,
            deltax2=box2,
            res_ndim=n_avgdim,
            boxlength=length,
            bins=n_psbins,
            bin_ave=True,
            get_variance=False,
            vol_normalised_power=True,
            log_bins=log_bins,
            k_weights=k_weights,
            remove_shotnoise=remove_shotnoise
        )
        k[0] = 0.
        return p_k,k

#chunk a lightcone into boxes and compute power spectra
def compute_power_lc(
        lc,
        redshifts,
        lc2=None,
        kind='brightness_temperature',
        kind2=None,
        n_psbins=None,
        log_bins=True,
        ignore_kperp_zero=True,
        ignore_kpar_zero=False,
        ignore_k_zero=False,
    ):
        if lc2 is None:
            lc2 = lc
        if kind2 is None:
            kind2 = kind

        dim = lc.shape

        if dim[0] != dim[1]:
            logger.error(f"bad LC shape {dim}")
            raise ValueError

        #find slices centred at desired redshifts
        slice_idx = np.argmin(np.fabs(lc.lightcone_redshifts[:,None] - redshifts[None,:]),axis=0)
        slice_idx = np.clip(slice_idx,dim[0]//2,dim[2]-dim[0]//2-dim[0]%2)
        z_slice = lc.lightcone_redshifts[slice_idx]

        #if dim[0] is odd, the centre is on slice_idx
        #if dim[0] is even, the centre is between slice_idx and slice_idx + 1
        slice_min = slice_idx - dim[0]//2
        slice_max = slice_idx + dim[0]//2 + dim[0]%2

        field_1 = getattr(lc,kind)
        field_2 = getattr(lc2,kind2)

        k_arr = np.zeros((len(redshifts),n_psbins))
        p_arr = np.zeros((len(redshifts),n_psbins))

        for i in range(len(redshifts)):
            slmin = slice_min[i]
            slmax = slice_max[i]
            logger.info(f'Starting ps {i} z {z_slice[i]} idx {slmin,slmax}')
            #get the power spectra
            p_arr[i,:],k_arr[i,:] = compute_power(field_1[:,:,slmin:slmax],
                                lc.user_params.BOX_LEN / lc.user_params.HII_DIM * np.array(field_1[:,:,slmin:slmax].shape), #chunk dimensions
                                n_psbins=n_psbins,
                                box2=field_2[:,:,slmin:slmax],
                                log_bins=log_bins,
                                ignore_kperp_zero=ignore_kperp_zero,
                                ignore_kpar_zero=ignore_kpar_zero,
                                ignore_k_zero=ignore_k_zero)

        return k_arr,p_arr,z_slice

def get_lc_powerspectra(lc_list,z_list,kind='brightness_temp',kind2=None,subtract_mean=[False,False],
                        divide_mean=[False,False],n_psbins=20,z_type='redshift'):
    '''
    Outputs 1D powerspectra from a list of lightcones.

    Parameters
    ----------
    lc_list: list of LightCone objects or filenames containing saved LightCone objects
    z_list: list of redshift targets at which to calculate the power spectrum
    kind: name of the first field
    kind2: name of the second field, None for auto-power
    subtract_mean: bool[2] whether to subtract the mean from each signal
    divide_mean: bool[2] whether to divide the mean from each signal
    n_psbins: number of power spectra bins

    Returns
    -------
    k_arr: (lightcone,redshift,k) array of k-bin centres
    power_arr: (lightcone,redshift,k) array of power in bin
    z_ps (lightcone,redshift): actual redshifts of the power spectra (closest slices to z_list)
    '''
    
    n_z = len(z_list)
    if isinstance(z_type,str):
        z_type = [z_type,]*n_z

    k_arr = np.zeros((len(lc_list),n_z,n_psbins))
    power_arr = np.zeros((len(lc_list),n_z,n_psbins))
    z_ps = np.zeros((len(lc_list),n_z))
    for i,lcf in enumerate(lc_list):
        if isinstance(lcf,p21c.LightCone):
            lc = lcf
        else:
            lc = p21c.LightCone.read(lcf,safe=False)

        z_targets = np.zeros_like(z_list)
        for j,(z,zt) in enumerate(zip(z_list,z_type)):
            if zt.lower() == 'redshift':
                z_targets[j] = z
            elif zt.lower() == 'xhi':
                z_targets[j] = np.interp(z,lc.global_xH[::-1],lc.node_redshifts[::-1])
            elif zt.lower() == 'bt_zero':
                #we want the reheating shift, not post-ion or re-coupling
                #interpolate between nodes by finding the last crossing of zero
                idx_heat = np.argwhere((np.diff(np.sign(lc.global_brightness_temp)) >= 1)).min()
                interp_range = lc.global_brightness_temp[idx_heat-10:idx_heat+10] #increasing
                z_targets[j] = np.interp(0,interp_range,lc.node_redshifts[idx_heat-10:idx_heat+10])
            elif zt.lower() == 'bt_min':
                z_targets[j] = lc.node_redshifts[np.argmin(lc.global_brightness_temp)]
            elif zt.lower() == 'bt_max':
                z_targets[j] = lc.node_redshifts[np.argmax(lc.global_brightness_temp)]
            else:
                raise ValueError(f"{zt} not a real ztype")
        logger.info(f"targets {z_list} ({z_type}) ==> {z_targets}")
        #subtract/divide the means in each slice
        field_1 = getattr(lc,kind)
        mean_z_1 = getattr(lc,kind).mean(axis=(0,1))[None,None,:]
        if subtract_mean[0]:
            field_1 = field_1 - mean_z_1
        if divide_mean[0]:
            field_1 = field_1 / mean_z_1
        setattr(lc,kind+'_ps',field_1)

        if kind2 is not None:
            field_2 = getattr(lc,kind2)
            mean_z_2 = getattr(lc,kind2).mean(axis=(0,1))[None,None,:]
            if subtract_mean[1]:
                field_2 = field_2 - mean_z_2
            if divide_mean[1]:
                field_2 = field_2 / mean_z_2
            setattr(lc,kind2+'_ps',field_2)

        logger.info(f'getting {len(z_targets)} powerspectra in {n_psbins} bins')

        k, power, z_ps[i,...] = compute_power_lc(lc,z_targets,kind=kind+'_ps',kind2=kind2+'_ps' if kind2 is not None else None,n_psbins=n_psbins)
        k_arr[i,...] = k
        power_arr[i,...] = power * k**3 / (2*np.pi**2) #dimensionless?
        
    return k_arr, power_arr, z_ps

def global_integrals(inputs,fields,ave_mturns,ave_mturns_mini,lnMmin,lnMmax):
    s_per_yr = 60*60*24 * 365.25

    ap_c = inputs.astro_params.cdict #since there are some values we need to access directly
    cp = inputs.cosmo_params
    up = inputs.user_params
    fo = inputs.flag_options
    redshifts = inputs.node_redshifts

    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, inputs.astro_params.cstruct, fo.cstruct)
    lib.init_ps()

    if up.INTEGRATION_METHOD_ATOMIC == 1 or up.INTEGRATION_METHOD_MINI == 1:
        lib.initialise_GL(lnMmin,lnMmax)

    lib.initialiseSigmaMInterpTable(np.exp(lnMmin), np.exp(lnMmax))
    
    # set the power-law limit masses
    if ap_c['ALPHA_STAR'] != 0.0:
        Mlim_Fstar = 1e10 * (ap_c['F_STAR10']) ** (-1.0 / ap_c['ALPHA_STAR'])
        Mlim_Fstar_MINI = 1e7 * (ap_c['F_STAR7_MINI']) ** (-1.0 / ap_c['ALPHA_STAR_MINI'])
    else:
        Mlim_Fstar = 0.0
        Mlim_Fstar_MINI = 0.0
    if ap_c['ALPHA_ESC'] != 0.0:
        Mlim_Fesc = 1e10 * (ap_c['F_ESC10']) ** (-1.0 / ap_c['ALPHA_ESC'])
        Mlim_Fesc_MINI = 1e7 * (ap_c['F_ESC7_MINI']) ** (-1.0 / ap_c['ALPHA_ESC'])
    else:
        Mlim_Fesc = 0.0
        Mlim_Fesc_MINI = 0.0
        
    rhocrit = (cp.cosmo.critical_density(0)).to('M_sun Mpc-3').value
    hubble = cp.cosmo.H(redshifts).to('s-1').value
    
    need_fesc_acg = any(field in ("n_ion","F_coll","whalo_sfr","xH_box") for field in fields)
    need_fesc_mcg = any(field in ("n_ion","F_coll_MINI","whalo_sfr","xH_box") for field in fields)
    need_star_acg = any(field in ("halo_xray","halo_stars","halo_sfr") for field in fields)
    need_star_mcg = any(field in ("halo_xray","halo_stars_mini","halo_sfr_mini") for field in fields)
    need_fcoll = any(field in ("halo_mass") for field in fields)
    need_xray = "halo_xray" in fields and fo.USE_HALO_FIELD

    if need_fesc_acg:
        fesc_acg_integral = np.vectorize(lib.Nion_General)(
            redshifts,lnMmin,lnMmax,
            ave_mturns,ap_c["ALPHA_STAR"],ap_c["ALPHA_ESC"],
            ap_c["F_STAR10"],ap_c["F_ESC10"],Mlim_Fstar,Mlim_Fesc
        )
    if need_fesc_mcg:
        fesc_mcg_integral = np.vectorize(lib.Nion_General_MINI)(
            redshifts,lnMmin,lnMmax,
            ave_mturns_mini,ave_mturns,ap_c["ALPHA_STAR_MINI"],ap_c["ALPHA_ESC"],
            ap_c["F_STAR7_MINI"],ap_c["F_ESC7_MINI"],
            Mlim_Fstar_MINI,Mlim_Fesc_MINI
        )
    if need_star_acg:
        star_acg_integral = np.vectorize(lib.Nion_General)(
            redshifts,lnMmin,lnMmax,
            ave_mturns,ap_c["ALPHA_STAR"],0.,
            ap_c["F_STAR10"],1.,Mlim_Fstar,0.
        )
    if need_star_mcg:
        star_mcg_integral = np.vectorize(lib.Nion_General_MINI)(
            redshifts,lnMmin,lnMmax,
            ave_mturns_mini,ave_mturns,ap_c["ALPHA_STAR_MINI"],0.,
            ap_c["F_STAR7_MINI"],0.,
            Mlim_Fstar_MINI,0.
        )
    if need_fcoll:
        fcoll_integral = np.vectorize(lib.Fcoll_General)(redshifts,lnMmin,lnMmax)
    if need_xray:
        xray_integral = np.vectorize(lib.Xray_General)(
            redshifts,lnMmin,lnMmax,
            ave_mturns_mini,ave_mturns,ap_c['ALPHA_STAR'],
            ap_c['ALPHA_STAR_MINI'],ap_c["F_STAR10"],ap_c["F_STAR7_MINI"],
            ap_c['L_X'],ap_c['L_X_MINI'],1/hubble,ap_c['t_STAR'],
            Mlim_Fstar,Mlim_Fstar_MINI,
        )

    results = []
    for field in fields:
        if field == 'halo_mass':
            result = fcoll_integral * cp.OMm * rhocrit
        elif field == 'halo_stars':
            result = star_acg_integral * cp.OMb * rhocrit * ap_c["F_STAR10"]
        elif field == 'halo_stars_mini':
            result = star_mcg_integral * cp.OMb * rhocrit * ap_c["F_STAR7_MINI"]
        elif field == 'halo_sfr':
            result = star_acg_integral * cp.OMb * rhocrit * ap_c["F_STAR10"] * hubble / ap_c["t_STAR"]
        elif field == 'halo_sfr_mini':
            result = star_mcg_integral * cp.OMb * rhocrit * ap_c["F_STAR7_MINI"] * hubble / ap_c["t_STAR"]
        elif field == 'n_ion':
            result = fesc_acg_integral * cp.OMb * rhocrit * ap_c["F_STAR10"] * p21c.global_params.Pop2_ion * ap_c["F_ESC10"]
            if fo.USE_MINI_HALOS:
                result += fesc_mcg_integral * cp.OMb * rhocrit * ap_c["F_STAR7_MINI"] * ap_c["F_ESC7_MINI"] * p21c.global_params.Pop3_ion
        elif field == 'whalo_sfr':
            result = fesc_acg_integral * cp.OMb * rhocrit * ap_c["F_STAR10"] * ap_c["F_ESC10"] * hubble / ap_c["t_STAR"] * p21c.global_params.Pop2_ion
            if fo.USE_MINI_HALOS:
                result += fesc_mcg_integral * cp.OMb * rhocrit * ap_c["F_STAR7_MINI"] * ap_c["F_ESC7_MINI"] * hubble / ap_c["t_STAR"] * p21c.global_params.Pop3_ion
        elif field == 'halo_xray':
            if lc.fo.USE_HALO_FIELD:
                result = star_acg_integral * cp.OMb * rhocrit * ap_c["F_STAR10"] * ap_c["L_X"] * hubble / ap_c["t_STAR"] * 1e-38 * s_per_yr
                if fo.USE_MINI_HALOS:
                    result += star_mcg_integral * cp.OMb * rhocrit * ap_c["F_STAR7_MINI"] * ap_c["L_X_MINI"] * hubble / ap_c["t_STAR"] * 1e-38 * s_per_yr
            else:
                result = xray_integral * cp.OMm * rhocrit * 1e-38
        elif field == 'xH_box':
            result = fesc_acg_integral * ap_c["F_STAR10"] * p21c.global_params.Pop2_ion * ap_c["F_ESC10"]
            if fo.USE_MINI_HALOS:
                result += fesc_mcg_integral * ap_c["F_STAR7_MINI"] * ap_c["F_ESC7_MINI"] * p21c.global_params.Pop3_ion
        elif field == 'Fcoll':
            result = fesc_acg_integral
        elif field == 'Fcoll_MINI':
                result += fesc_mcg_integral
        else:
            raise ValueError("bad field")

        results += [result]
    
    return np.array(results)

def match_global_function(fields,lc,lnMmin,lnMmax):
    """
    Gets the expected global average matching a HaloBox field.

    uses the lightcone object to get the necessary parameters
    """

    inputs = p21c.InputParameters(
        astro_params = lc.astro_params,
        cosmo_params = lc.cosmo_params,
        user_params = lc.user_params,
        flag_options = lc.flag_options,
        redshifts = lc.node_redshifts,
        random_seed = lc.random_seed,
    )
    
    ave_mturns = 10**lc.log10_mturnovers
    ave_mturns_mini = 10**lc.log10_mturnovers_mini

    return global_integrals(inputs,fields,ave_mturns,ave_mturns_mini,lnMmin,lnMmax)