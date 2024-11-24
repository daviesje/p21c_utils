#downsample a 3d array by a factor
import numpy as np
from scipy.stats import binned_statistic as binstat
from powerbox.tools import get_power
from copy import deepcopy

import py21cmfast as p21c
from py21cmfast.c_21cmfast import ffi,lib
from astropy import units as U

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Generates fake ionized, spintemp and perturbed halo fields for getting fields w/o feedback
def get_fake_boxes(kw_dict,random_seed):
    #fake ion field
    fake_ion = p21c.IonizedBox(
            **kw_dict,
            random_seed=random_seed
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
            **kw_dict,
            random_seed=random_seed
    )
    fake_ts() #allocate memory
    fake_ts.J_21_LW_box[...] = 0.
    fake_ts() #pass to C
    
    for k, state in fake_ts._array_state.items():
            if state.initialized:
                    state.computed_in_mem = True
    
    #fake pt halos for subsampler box
    fake_pth = p21c.PerturbHaloField(
            **kw_dict,
            random_seed=random_seed,
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
def get_halo_fields(zstart,zstop,zstep,random_seed,cp,up,ap,fo,init_box,ptb):
    z = zstop
    zst = z if zstart is None else zstart
    while z > zst:
        z = (1+z)/zstep - 1
    
    halos_desc = None
    while z <= zstop:
        kw = {
            "redshift": z,
            "cosmo_params" : cp,
            "user_params" : up,
            "astro_params" : ap,
            "flag_options" : fo,
        }
        halo_field = p21c.determine_halo_list(
            **kw,
            init_boxes=init_box,
            regenerate=True,
            halos_desc=halos_desc,
            random_seed=random_seed,
        )
        logger.info(f'z = {z} zi = {zst} zf = {zstop}')
        halos_desc = halo_field
        z = (1+z)*p21c.global_params.ZPRIME_STEP_FACTOR - 1
    
    z = (1+z)/p21c.global_params.ZPRIME_STEP_FACTOR - 1
    pt_halos = p21c.perturb_halo_list(
        **kw,
        init_boxes=init_box,
        halo_field=halo_field,
        regenerate=True,
    )
    fts,fion,fpth = get_fake_boxes(**kw)
    hbox_ss = p21c.make_halo_box(
        **kw,
        init_boxes=init_box,
        regenerate=True,
        pt_halos=fpth,
        perturbed_field=ptb,
        previous_spin_temp=fts,
        previous_ionize_box=fion,
    )
    return halo_field,pt_halos,hbox_ss

#gets properties from either HaloField or PerturbHaloField
#ignoring feedback and with possible alterations to parameters
def get_props_from_halofield(halo_field,ap_alter=None,fo_alter=None,sel=None,kinds=['sfr',]):
    #fake pt_halos
    zero_array = ffi.cast("float *", np.zeros(1).ctypes.data)

    if sel is None:
        sel = slice(0,halo_field.n_halos+1)
    n_halos = halo_field.halo_masses[sel].size

    pt_halos = p21c.PerturbHaloField(
                    redshift=halo_field.redshift,
                    user_params=halo_field.user_params,
                    cosmo_params=halo_field.cosmo_params,
                    astro_params=halo_field.astro_params,
                    flag_options=halo_field.flag_options,
                    buffer_size=n_halos
            )
    pt_halos()
    pt_halos.n_halos = n_halos
    pt_halos.halo_masses[...] = halo_field.halo_masses[sel]
    pt_halos.star_rng[...] = halo_field.star_rng[sel]
    pt_halos.sfr_rng[...] = halo_field.sfr_rng[sel]
    pt_halos.xray_rng[...] = halo_field.xray_rng[sel]

    ap = halo_field.astro_params if ap_alter is None else ap_alter
    fo = halo_field.flag_options if fo_alter is None else fo_alter

    #get props w/o feedback
    props_out = np.zeros(int(12*pt_halos.n_halos)).astype('f4')
    lib.test_halo_props(
            halo_field.redshift,
            halo_field.user_params(),
            halo_field.cosmo_params(),
            ap(),
            fo(),
            zero_array,
            zero_array,
            zero_array,
            zero_array,
            pt_halos(),
            ffi.cast("float *", props_out.ctypes.data),
    )

    props_out = props_out.reshape((pt_halos.n_halos,12))
    index_map = {'mass' : 0, 'star' : 1, 'sfr' : 2, 'xray' : 3, 'nion' : 4, 'wsfr' : 5,
                 'star_mini' : 6, 'sfr_mini' : 7, 'mturn_a' : 8, 
                 'mturn_m' : 9, 'mturn_r' : 10, 'metallicity' : 11}
    
    return [props_out[:,index_map[kind]] for kind in kinds]

def get_cell_integrals(redshift,user_params,cosmo_params,astro_params,flag_options,delta):
    #pass info to backend
    lib.Broadcast_struct_global_all(user_params(), cosmo_params(), astro_params(), flag_options())
    lib.init_ps()
    if user_params.INTEGRATION_METHOD_ATOMIC == 1:
        lib.initialise_GL(100, lnMmin, lnMmax)

    rhocrit = cosmo_params.cosmo.critical_density(0)
    growth = lib.dicke(redshift)
    lnMmin = np.log(user_params.SAMPLER_MIN_MASS)
    lnMmax = np.log(1e15) #must be > cell but < table max
    cell_volume = (user_params.BOX_LEN * U.Mpc / user_params.HII_DIM) ** 3
    cell_mass = (rhocrit * cosmo_params.OMm * cell_volume).to("M_sun")
    sigma_cell = lib.sigma_z0(cell_mass.value)
    Mlim_Fstar = 1e10 * (10**astro_params.F_STAR10) ** (-1.0 / astro_params.ALPHA_STAR)
    Mlim_Fesc = 1e10 * (10**astro_params.F_ESC10) ** (-1.0 / astro_params.ALPHA_ESC)
    
    m_integral = lib.Mcoll_Conditional(
            growth,
            lnMmin,
            lnMmax,
            cell_mass.value,
            sigma_cell,
            delta,
            0,
        )
    integral_starsonly = lib.Nion_ConditionalM(
        growth,
        lnMmin,
        lnMmax,
        cell_mass.value,
        sigma_cell,
        delta,
        10**astro_params.M_TURN,
        astro_params.ALPHA_STAR,
        0.,
        10**astro_params.F_STAR10,
        1.,
        Mlim_Fstar,
        0.,
        user_params.INTEGRATION_METHOD_ATOMIC,
    )
    integral_fesc = lib.Nion_ConditionalM(
        growth,
        lnMmin,
        lnMmax,
        cell_mass.value,
        sigma_cell,
        delta,
        10**astro_params.M_TURN,
        astro_params.ALPHA_STAR,
        astro_params.ALPHA_ESC,
        10**astro_params.F_STAR10,
        10**astro_params.F_ESC10,
        Mlim_Fstar,
        Mlim_Fesc,
        user_params.INTEGRATION_METHOD_ATOMIC,
    )

    #TODO: add minihalos

    return m_integral,integral_starsonly,integral_fesc


#gets expected HaloBox Values given a cell delta.
#  Should be equivalent to calling backend (set_fixed_grids)
#  on one cell w/o feedback
def get_expected_lagrangian(redshift,user_params,cosmo_params,astro_params,flag_options,delta):

    lib.initialiseSigmaMInterpTable(user_params.SAMPLER_MIN_MASS, 1e16)

    #setup cell values & constants
    t_h = (1 / cosmo_params.cosmo.H(z)).to('s-1')
    s_per_yr = 60 * 60 * 24 * 365.25

    rhocrit = cosmo_params.cosmo.critical_density(0)
    prefactor_mass = rhocrit * cosmo_params.OMm
    prefactor_stars = rhocrit * cosmo_params.OMb * astro_params.F_STAR10
    prefactor_sfr = prefactor_stars / astro_params.t_star / t_h
    prefactor_nion = prefactor_stars * astro_params.fesc_10 * p21c.global_params.Pop2_ion
    prefactor_wsfr = prefactor_sfr * astro_params.fesc_10
    prefactor_xray = prefactor_sfr * astro_params.l_x * s_per_yr

    #TODO: add minihalos
    # prefactor_stars_mini = rhocrit * cosmo_params.OMb * astro_params.F_STAR7
    # prefactor_sfr_mini = prefactor_stars_mini / astro_params.t_star / t_h
    # prefactor_nion_mini = prefactor_stars_mini * astro_params.fesc_7 * p21c.global_params.Pop3_ion
    # prefactor_wsfr_mini = prefactor_sfr_mini * astro_params.fesc_7
    # prefactor_xray_mini = prefactor_sfr_mini * astro_params.l_x_mini * s_per_yr

    m_int, s_int, f_int = get_cell_integrals(redshift,user_params,cosmo_params,astro_params,flag_options,delta)

    results = {
        'halo_mass': m_int * prefactor_mass,
        'halo_stars' : s_int * prefactor_stars,
        'halo_sfr' : s_int * prefactor_sfr,
        'n_ion' : f_int * prefactor_nion,
        'fescweighted_sfr' : f_int * prefactor_wsfr,
        'halo_xray' : s_int * prefactor_xray,
    }

    return results

def get_expected_eulerian(redshift,user_params,cosmo_params,astro_params,flag_options,delta,convert=False):
    #Mo & White Conversion
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
def grid_halo_cat(halo_cat,user_params):
    lores = user_params.HII_DIM
    hires = user_params.DIM
    cell_volume = (user_params.BOX_LEN * U.Mpc / user_params.HII_DIM) ** 3
    bins_crd = np.linspace(-0.5,lores+0.5,num=lores+1)
    masses = getattr(halo_cat,'halo_masses')
    crd = getattr(halo_cat,'halo_coords')
    crd = crd.reshape(masses.size,3)
    crd = crd / hires * lores


    nion,xray =  get_props_from_halofield(halo_cat,kinds=['nion','xray'])
    logger.info(f'doing hist crd {crd.shape} props ({masses.shape},{nion.shape},{xray.shape}) bins {bins_crd.shape}')
    mass_grid,_ = np.histogramdd(crd,bins=[bins_crd,]*3,weights=masses)
    nion_grid,_ = np.histogramdd(crd,bins=[bins_crd,]*3,weights=nion)
    xray_grid,_ = np.histogramdd(crd,bins=[bins_crd,]*3,weights=xray)
    mass_grid = mass_grid * U.M_sun / cell_volume
    nion_grid = nion_grid / cell_volume
    xray_grid = xray_grid  * U.erg / U.s / cell_volume

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
                z_targets[j] = np.interp(z,lc.global_xHI[::-1],lc.node_redshifts[::-1])
            elif zt.lower() == 'bt_min':
                z_targets[j] = lc.node_redshifts[np.argmin(lc.global_brightness_temp)]
            elif zt.lower() == 'bt_max':
                z_targets[j] = lc.node_redshifts[np.argmax(lc.global_brightness_temp)]
            else:
                raise ValueError(f"{zt} not a real ztype")
        logger.info(f"XHI targets {z_list} ({z_type}) ==> {z_targets}")
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
        power_arr[i,...] = power * k**3 / (2*np.pi) #dimensionless?
        
    return k_arr, power_arr, z_ps

def match_global_function(fields,lc,**kwargs):
    """
    Gets the expected global average matching a HaloBox field.
    """
    astro_params = lc.astro_params
    cosmo_params = lc.cosmo_params
    user_params = lc.user_params
    flag_options = lc.flag_options
    redshifts = lc.node_redshifts

    s_per_yr = 60*60*24 * 365.25
    
    lib.Broadcast_struct_global_all(
        user_params.cstruct,
        cosmo_params.cstruct,
        astro_params.cstruct,
        flag_options.cstruct
    )
    lib.init_ps()

    if user_params.INTEGRATION_METHOD_ATOMIC == 1 or user_params.INTEGRATION_METHOD_MINI == 1:
        lib.initialise_GL(kwargs["lnMmin"], kwargs["lnMmax"])

    lib.initialiseSigmaMInterpTable(kwargs["Mmin"], kwargs["Mmax"])
    
    ave_mturns = 10**lc.log10_mturnovers
    ave_mturns_mini = 10**lc.log10_mturnovers_mini
    
    # set the power-law limit masses
    if astro_params.ALPHA_STAR != 0.0:
        Mlim_Fstar = 1e10 * (10**astro_params.F_STAR10) ** (-1.0 / astro_params.ALPHA_STAR)
        Mlim_Fstar_MINI = 1e7 * (10**astro_params.F_STAR7_MINI) ** (-1.0 / astro_params.ALPHA_STAR_MINI)
    else:
        Mlim_Fstar = 0.0
        Mlim_Fstar_MINI = 0.0
    if astro_params.ALPHA_ESC != 0.0:
        Mlim_Fesc = 1e10 * (10**astro_params.F_ESC10) ** (-1.0 / astro_params.ALPHA_ESC)
        Mlim_Fesc_MINI = 1e7 * (10**astro_params.F_ESC7_MINI) ** (-1.0 / astro_params.ALPHA_ESC)
    else:
        Mlim_Fesc = 0.0
        Mlim_Fesc_MINI = 0.0
        
    rhocrit = (cosmo_params.cosmo.critical_density(0)).to('M_sun Mpc-3').value
    hubble = cosmo_params.cosmo.H(redshifts).to('s-1').value
    
    need_fesc_acg = field in ("n_ion","F_coll","whalo_sfr","xH_box")
    need_fesc_mcg = field in ("n_ion","F_coll_MINI","whalo_sfr","xH_box")
    need_star_acg = field in ("halo_xray","halo_stars","halo_sfr")
    need_star_mcg = field in ("halo_xray","halo_stars_mini","halo_sfr_mini")
    need_fcoll = field in ("halo_mass")

    ap_c = astro_params.cdict

    if need_fesc_acg:
        fesc_acg_integral = np.vectorize(lib.Nion_General)(
            redshifts,kwargs['lnMmin'],kwargs['lnMmax'],
            ave_mturns,ap_c["ALPHA_STAR"],ap_c["ALPHA_ESC"],
            ap_c["F_STAR10"],ap_c["F_ESC10"],Mlim_Fstar,Mlim_Fesc
        )
    if need_fesc_mcg:
        fesc_mcg_integral = np.vectorize(lib.Nion_General_MINI)(
            redshifts,kwargs['lnMmin'],kwargs['lnMmax'],
            ave_mturns_mini,ave_mturns,ap_c["ALPHA_STAR_MINI"],ap_c["ALPHA_ESC"],
            ap_c["F_STAR7_MINI"],ap_c["F_ESC7_MINI"],
            Mlim_Fstar_MINI,Mlim_Fesc_MINI
        )
    if need_fesc_acg:
        star_acg_integral = np.vectorize(lib.Nion_General)(
            redshifts,kwargs['lnMmin'],kwargs['lnMmax'],
            ave_mturns,ap_c["ALPHA_STAR"],0.,
            ap_c["F_STAR10"],1.,Mlim_Fstar,0.
        )
    if need_fesc_mcg:
        star_mcg_integral = np.vectorize(lib.Nion_General_MINI)(
            redshifts,kwargs['lnMmin'],kwargs['lnMmax'],
            ave_mturns_mini,ave_mturns,ap_c["ALPHA_STAR_MINI"],0.,
            ap_c["F_STAR7_MINI"],0.,
            Mlim_Fstar_MINI,0.
        )
    if need_fcoll:
        fcoll_integral = np.vectorize(lib.Fcoll_General)(redshifts,kwargs['lnMmin'],kwargs['lnMmax'])

    results = []
    for field in fields:
        if field == 'halo_mass':
            result = fcoll_integral * cosmo_params.OMm * rhocrit
        elif field == 'halo_stars':
            result = star_acg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR10"]
        elif field == 'halo_stars_mini':
            result = star_mcg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR7_MINI"]
        elif field == 'halo_sfr':
            result = star_acg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR10"] * hubble / ap_c["t_STAR"]
        elif field == 'halo_sfr_mini':
            result = star_mcg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR7_MINI"] * hubble / ap_c["t_STAR"]
        elif field == 'n_ion':
            result = fesc_acg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR10"] * p21c.global_params.Pop2_ion * ap_c["F_ESC10"]
            if flag_options.USE_MINI_HALOS:
                result += fesc_mcg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR7_MINI"] * ap_c["F_ESC7_MINI"] * p21c.global_params.Pop3_ion
        elif field == 'whalo_sfr':
            result = fesc_acg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR10"] * ap_c["F_ESC10"] * hubble / ap_c["t_STAR"] * p21c.global_params.Pop2_ion
            if flag_options.USE_MINI_HALOS:
                result += fesc_mcg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR7_MINI"] * ap_c["F_ESC7_MINI"] * hubble / ap_c["t_STAR"] * p21c.global_params.Pop3_ion
        elif field == 'halo_xray':
            result = star_acg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR10"] * ap_c["L_X"] * hubble / ap_c["t_STAR"] * 1e-38 * s_per_yr
            if flag_options.USE_MINI_HALOS:
                result += star_mcg_integral * cosmo_params.OMb * rhocrit * ap_c["F_STAR7_MINI"] * ap_c["L_X_MINI"] * hubble / ap_c["t_STAR"] * 1e-38 * s_per_yr
        elif field == 'xH_box':
            result = fesc_acg_integral * ap_c["F_STAR10"] * p21c.global_params.Pop2_ion * ap_c["F_ESC10"]
            if flag_options.USE_MINI_HALOS:
                result += fesc_mcg_integral * ap_c["F_STAR7_MINI"] * ap_c["F_ESC7_MINI"] * p21c.global_params.Pop3_ion
        elif field == 'Fcoll':
            result = fesc_acg_integral
        elif field == 'Fcoll_MINI':
                result += fesc_mcg_integral
        else:
            raise ValueError("bad field")

        results += result
    
    return np.array(result)
