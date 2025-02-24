#Useful functions stolen from various places

import numpy as np
from powerbox.tools import get_power
import py21cmfast as p21c
from astropy import units as U, constants as C
from scipy import integrate
from scipy.ndimage import gaussian_filter

from .noise_models import add_bt_noise, add_cii_noise
from .analysis_funcs import get_lc_powerspectra
from .spec_conversions import abs_to_app, sfr_to_Muv
from astropy.cosmology import z_at_value

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(ch)

HI_NU_ION = 13.6
HeI_NU_ION = 24.6
HeII_NU_ION = 54.4
X_H = 0.76
Y_He = 1 - X_H
Y_He_N = 0.25*Y_He/(X_H + 0.25*Y_He)
X_H_N = X_H/(X_H + 0.25*Y_He)

def crosssec_HI(nu):
    Z = 1
    if (nu < HI_NU_ION):
        return 0

    nu_in = nu
    if (nu == HI_NU_ION):
        nu_in += 1e-12

    epsilon = np.sqrt(nu_in/HI_NU_ION - 1)
    return (6.3e-18)/Z/Z * (HI_NU_ION/nu_in)**4 * np.exp(4-(4*np.atan(epsilon)/epsilon)) / (1-np.exp(-2*np.pi/epsilon))

def crosssec_HeI(nu):
    if (nu < HeI_NU_ION):
        return 0

    x = nu/13.61 - 0.4434
    y = np.sqrt(x*x + 2.136**2)
    return  9.492e-16*((x-1)*(x-1) + 2.039*2.039) * y**(0.5 * 3.188 - 5.5) * (1.0 + np.sqrt(y/1.469))**-3.188

def crosssec_HeII(nu):
    Z = 2
    if (nu < HeII_NU_ION):
        return 0

    nu_in = nu
    if (nu == HeII_NU_ION):
        nu_in += 1e-12

    epsilon = np.sqrt(nu_in/HeII_NU_ION - 1)
    return (6.3e-18)/Z/Z * (HeII_NU_ION/nu_in)**4 * np.exp(4-(4*np.atan(epsilon)/epsilon)) / (1-np.exp(-2*np.pi/epsilon))

# def tau_n_integrand(z,z0,nu0,Q,xH_ntl,cosmo):
#     hterm = C.c / (1+z) / cosmo.H(z)
#     nu = nu0*(1+z)/(1+z0)
#     sigma = X_H_N*(xH_ntl)*crosssec_HI(nu) + Y_He_N*(xH_ntl)*crosssec_HeI(nu) + Y_He_N*(1-xH_ntl)*crosssec_HeII(nu)
#     return hterm * (1-Q) * cosmo.critical_density(0) * (1+z)**3 * sigma

# def tau_n(z_arr,z0,nu0,Q_z,xH_z,cosmo):
#     integrand = tau_n_integrand(z_arr,z0,nu0,Q_z,xH_z,cosmo)
#     result = integrate.trapz(integrand,z_arr,axis=-1)

#     return result

# def tau_i_integrand(z,z0,nu0):
#     A = 1.
#     alpha = 1.
#     beta = 1. #TODO: set the constants
#     N_HeII = 1.

#     nu = nu0*(1+z)/(1+z0)

#     nhterm = A*N_HI**alpha*(1+z)**beta * (1 - np.exp(-(N_HI*crosssec_HI(nu) + N_HeII*crosssec_HeII(nu))))
#     return

# def tau_i(z_arr,z0,nu0,Q_z,xH_z,cosmo):
#     integrand = np.zeros_like(z_arr)
#     for z in z_arr:
#         integrand[i] = tau_i_integrand(z_arr,z0,nu0,Q_z,xH_z,cosmo)
#     result = integrate.trapz(integrand,z_arr,axis=-1)

#     return result

# OUT OF DATE!!! REDO WITH HALOBOX XRAY!
def calc_xray_lc(nu,cosmo_lc,up_lc,ap_lc,z_lc,z_obs_lc,sfr_lc):
    if not hasattr(nu,'unit'):
        nu = nu * U.eV
    # hplanck = 6.62606896e-27 # erg s
    # NU_over_EV = 1.60217646e-12 / hplank # erg eV-1 erg-1 s-1 -> (eV s)-1
    prefactor = (1+z_obs_lc)**3 / (4*np.pi) * C.c
    cellvol = (1+z_lc)**-3 * (up_lc.BOX_LEN/up_lc.HII_DIM * U.Mpc)**3
    
    #xh_lc = getattr(lc,'xH_box')
    #Q_HI_LC = xh_lc > 0.1
    #Q_LC_mean = np.average(Q_HI_LC,axis=(0,1)) #TODO: check threshold
    #xH_ntl_mean = np.average(xh_lc,weight=Q_HI_LC,axis=(0,1))

    # sfr_lc = getattr(lc,'halo_sfr') * U.M_sun / U.year
    # logger.info(f'reading z')
    #z_lc = lc.lightcone_redshifts[None,None,:]

    # sel = z_lc[0,0,:] > z
    # z_lc = z_lc[...,sel]
    # sfr_lc = sfr_lc[...,sel]

    #NU LIMITS FOR DETERMINING NORMALISATION
    nu_min = ap_lc.NU_X_THRESH * U.eV
    nu_max = p21c.global_params.NU_X_BAND_MAX * U.eV
    xa_lc = ap_lc.X_RAY_SPEC_INDEX

    if xa_lc == 1:
        L_factor = nu_min * np.log(nu_max/nu_min)
        L_factor = 1/L_factor
    else:
        L_factor = (nu_max)**(1-xa_lc) - (nu_min)**(1-xa_lc)
        L_factor = 1/L_factor
        L_factor *= (nu_min)**(-xa_lc) * (1 - xa_lc)

    # L_factor *= (3.1556226e7)/(hplank) #? (erg s)-1
    L_X = ap_lc.convert("L_X",ap_lc.L_X) * U.Unit('erg s−1 M_sun−1 yr')
    eps_x_over_sfrd = L_X * L_factor * (nu*(1+z_lc)/(1+z_obs_lc)/nu_min)**(-xa_lc)
    eps_x = sfr_lc * eps_x_over_sfrd
    eps_x = eps_x/(cosmo_lc.H(z_lc)*(1+z_lc)) * prefactor / cellvol

    #TODO: tau integral
    #compute_tau(blahblah)

    #logger.info(f'before integral: {eps_x.min()}, {eps_x.mean()}, {eps_x.max()}')
    #logger.info(f'Integrating')
    result = integrate.trapz(eps_x,z_lc,axis=-1)
    # logger.info(f'after integral: {result.min()}, {result.mean()}, {result.max()}')
    #,equivalencies=U.spectral()
    return result.to('erg s-1 eV-1 cm-2').value

# OUT OF DATE!!! REDO WITH HALOBOX XRAY!
def xray_bg(lc,z_obs=0,min_obs=500,max_obs=2000):
    sfr_lc = getattr(lc,'halo_sfr') * U.M_sun / U.year
    z_lc = lc.lightcone_redshifts[None,None,:]

    
    sel = z_lc[0,0,:] > z_obs
    z_lc = z_lc[...,sel]
    sfr_lc = sfr_lc[...,sel]

    ap_lc = lc.astro_params
    up_lc = lc.user_params
    cosmo_lc = lc.cosmo_params.cosmo
    z_obs_lc = z_obs

    res,err = integrate.quad_vec(calc_xray_lc,min_obs,max_obs,args=(cosmo_lc,up_lc,ap_lc,z_lc,z_obs_lc,sfr_lc))
    return res,err

def make_cii_map(lc):
    NU_CII = 1900.54 * U.GHz

    #DL2014 relation
    z_lc = lc.lightcone_redshifts[None,None,:]
    nu_lc = NU_CII / (1+z_lc)
    dz_lc = -(z_lc - np.concatenate((z_lc[...,1:],np.zeros((1,1,1))),axis=-1)) #NOTE:last cell wrong

    sfr = (lc.user_params.BOX_LEN/lc.user_params.HII_DIM)**3 * lc.halo_sfr * U.Unit('M_sun s-1')
    #De Looze 2014
    L_conversion = 10**7.06 * U.Unit('solLum M_sun-1 yr')
    L = sfr * L_conversion
    
    d_lum = lc.cosmo_params.cosmo.luminosity_distance(z_lc)
    mu = 1 #no magnification

    #EQ 3 Bethermin+ 2022
    I_conversion = mu /(1.04e-3 * U.Unit('Mpc-2 GHz-1') * nu_lc * d_lum**2) * U.Unit('Jy km s-1 solLum-1')
    I = I_conversion * L

    #angular size of a cell
    d_a = lc.cosmo_params.cosmo.angular_diameter_distance(z_lc)
    ang_size = ((lc.user_params.BOX_LEN/lc.user_params.HII_DIM) / (1+z_lc) * U.Mpc / d_a) * U.rad
    solid_angle = ang_size**2

    #EQ 11 Bethermin+ 2022
    B = (1+z_lc) / dz_lc / C.c / solid_angle * I
    B = B.to('Jy sr-1').value

    #surface brightness density (Jy/sr)
    return B,z_lc[0,0,:],nu_lc[0,0,:]

def make_cii_coev(cv):
    NU_CII = 1900.54 * U.GHz
    cosmo = cv.cosmo_params.cosmo
    cell_len = (cv.user_params.BOX_LEN/cv.user_params.HII_DIM)

    #DL2014 relation
    z = cv.redshift
    nu = NU_CII / (1+z)

    sfr = (cell_len)**3 * cv.halo_sfr * U.Unit('M_sun s-1')
    #De Looze 2014
    L_conversion = 10**7.06 * U.Unit('solLum M_sun-1 s')
    L = sfr * L_conversion
    
    d_lum = cosmo.luminosity_distance(z)
    mu = 1 #no magnification

    #EQ 3 Bethermin+ 2022
    #the strange unit conversion is from the constants, TODO: make it properly
    I_conversion = mu / 1.04e-3 * U.Unit('Jy km s-1 Mpc2 GHz-1 solLum-1')
    I = I_conversion / d_lum**2 * nu * L

    #angular size of a cell
    d_a = cosmo.angular_diameter_distance(z)
    ang_size = (cell_len / (1+z) * U.Mpc / d_a) * U.rad
    solid_angle = ang_size**2

    #EQ 11 Bethermin+ 2022
    dc_z = cosmo.comoving_distance(z)
    #Each cell is at the same z
    dz = z_at_value(cosmo.comoving_distance, dc_z + cell_len*U.Unit('Mpc')) - z
    B = (1+z) / dz / C.c / solid_angle * I
    B = B.to('Jy sr-1').value

    return B
    

def cii_xps(lc,
            z_max=50,
            bt_thermal=True,
            bt_uvcov=True,
            bt_wedge="boxcar",
            n_psbins=24,
            ps_z=np.array([8,10,12]),
            cii_noisefile=None,
            seed=1234,
            cii_hours=1):

    z_lc = lc.lightcone_redshifts
    cii_lc,_,nu_lc = make_cii_map(lc)

    cii_noise_info = None
    print(f'CII Before noise ({cii_lc.min()},{cii_lc.max()},{cii_lc.mean()})')
    if isinstance(cii_noisefile,str):
        print(f'using noise file {cii_noisefile}')
        cii_lc,cii_noise_info = add_cii_noise(cii_lc,nu_lc,lc,noisefile=cii_noisefile,seed=seed,hours=cii_hours)
        print(f'CII After noise ({cii_lc.min()},{cii_lc.max()},{cii_lc.mean()})')
        
    z_sel = z_lc < z_max
    cii_lc = cii_lc[...,z_sel]
    nu_lc = nu_lc[z_sel]

    z_lc = z_lc[z_sel]
    
    bt_lc = lc.brightness_temp[...,z_sel]
    bt_lc,_,_ = add_bt_noise(bt_lc,lc,z_lc,wedge=bt_wedge,thermal=bt_thermal,uvcov=bt_uvcov,seed=seed)

    setattr(lc,'CII_box',cii_lc)
    setattr(lc,'brightness_temp_diff',bt_lc)

    k_arr, power_arr, z_ps = get_lc_powerspectra([lc,],ps_z,kind='brightness_temp_diff',kind2='CII_box',n_psbins=n_psbins,
                                                  subtract_mean=[False,True],divide_mean=[False,True])
    
    return bt_lc, cii_lc, k_arr[0], power_arr[0], z_ps[0], cii_noise_info

#From a lightcone file, find cached halo fields, trim them and build a galaxy lightcone
def make_gal_lc(lc,muv_cut=20):
    z_lc = lc.lightcone_redshifts

    init_box = p21c.initial_conditions(
        user_params=lc.user_params,
        cosmo_params=lc.cosmo_params,
        random_seed=lc.random_seed,
        regenerate = False,
    )

    fnc = p21c.interp_functions.get("halo_sfr", "mean")

    lc_out = np.zeros_like(lc.halo_sfr)

    box_index = 0
    lc_index = 0
    halos_desc = None
    #make sure this is backwards in time!!!!
    for i,z in enumerate(z_lc[::-1]):
        halos = p21c.determine_halo_list(
                redshift=z,
                init_boxes=init_box,
                user_params=lc.user_params,
                cosmo_params=lc.cosmo_params,
                astro_params=lc.astro_params,
                flag_options=lc.flag_options,
                halos_desc=halos_desc,
                regenerate=False,
        )
        halos_desc = halos
        pt_halos = p21c.perturb_halo_list(
            redshift=z,
            init_boxes=init_box,
            cosmo_params=lc.cosmo_params,
            user_params=lc.user_params,
            astro_params=lc.astro_params,
            flag_options=lc.flag_options,
            halo_field=halos,
            regenerate=False,
        )

        sfr = halos.halo_sfr

        muv = abs_to_app(sfr_to_Muv(sfr))
        sel = muv > muv_cut

        logger.info(f'{sel.sum()} out of {sel.size} galaxies selected')

        #HACK: remove halos in the boxes without computing or changing n_halos
        halos.halo_mass[sel] = 0
        halos.stellar_mass[sel] = 0
        halos.halo_sfr[sel] = 0

        box = p21c.halo_box(redshift=z,
                    cosmo_params=lc.cosmo_params,
                    user_params=lc.user_params,
                    astro_params=lc.astro_params,
                    flag_options=lc.flag_options,
                    regenerate=False,
                    pt_halos=pt_halos,
                    perturbed_field=None,
        )

        if i > 0:
            n = p21c._interpolate_in_redshift(
                i,
                box_index,
                lc_index,
                z_lc.size,
                lc.lightcone_coords,
                lc.lc_distances,
                box,
                box_prev,
                "halo_sfr",
                lc_out,
                fnc,
            )
            lc_index += n
            box_index += n


        box_prev = halos

    return sfr_to_Muv(lc_out)

def gal_xps(lc,muv_cut=20,bt_thermal=True,bt_uvcov=True,bt_wedge="boxcar",n_psbins=24,ps_z=np.array([8,10,12])):
    gal_lc = make_gal_lc(lc)

    z_lc = lc.lightcone_redshifts
    bt_lc = lc.brightness_temp
    
    #meandens brightness temp
    #bt_mean = bt_lc.mean(axis=(0,1))
    #Zaldarriga+ 2004
    bt_mean = 23.8*np.sqrt((1+z_lc)/10)

    ps_idx = np.argmin(np.fabs(z_lc[None,:] - ps_z[:,None]),axis=1)
    
    bt_lc,_,_ = add_bt_noise(bt_lc,lc,z_lc,wedge=bt_wedge,thermal=bt_thermal,uvcov=bt_uvcov) * U.mK

    chunk_size = lc.user_params.HII_DIM
    results = []
    for i,z in enumerate(ps_z):
        chunk_min = int(ps_idx[i] - chunk_size//2)
        chunk_max = int(ps_idx[i] + chunk_size//2)
        # print(f'plotting {chunk_min}, {chunk_max}')
        bt_chunk = bt_lc[:,:,chunk_min:chunk_max]
        bt_chunk = bt_chunk / bt_mean[ps_idx[i]]
        gal_chunk = gal_lc[:,:,chunk_min:chunk_max]
        gal_chunk = gal_chunk / gal_chunk.mean() - 1

    return bt_lc, gal_lc, results

#ATHENA, 100^2 DEG FOV, 5'' ANGRES, 10^-16 FLUX LIMIT (erg cm^-2 s^-1)
def xray_xps(lc,z_bt=12,depth_bt=10,noise_bt=True,noise_x=1e-16,res_x_arcsec=5,n_psbins=20):
    z_lc = lc.lightcone_redshifts
    d_lc = lc.lightcone_distances

    xrb = xray_bg(lc)
    
    #find the minimum angular diameter distance, <ASSUME THIS IS CONSTANT?????>
    min_z = lc.lightcone_redshifts.min()
    d_a = lc.cosmo_params.cosmo.angular_diameter_distance(min_z)
    ang_size = ((lc.user_params.BOX_LEN/lc.user_params.HII_DIM) / (1+min_z) * U.Mpc / d_a) * U.rad
    res_rad = (res_x_arcsec * U.arcsec).to('rad')
    gauss_sigma = (res_rad/ang_size).to('').value
    print(f'sigma = {gauss_sigma}')

    x_lc,err = xray_bg(lc)
    logger.info(f'DONE XRAY')

    x_lc = gaussian_filter(x_lc,sigma=gauss_sigma,mode='wrap')
    random_field = np.random.normal(loc=0,scale=noise_x,size=x_lc.shape)
    x_lc = x_lc + random_field

    centre_idx = np.argmin(np.fabs(z_lc-z_bt))
    min_idx = np.argmin(np.fabs(d_lc-(d_lc[centre_idx]-depth_bt/2)))
    max_idx = np.argmin(np.fabs(d_lc-(d_lc[centre_idx]+depth_bt/2)))
    logger.info(f'plotting indices (z={z_bt},i={centre_idx}) ({min_idx}, {max_idx}) d=({d_lc[min_idx]},{d_lc[max_idx]})')
    
    bt_lc = getattr(lc,'brightness_temp')
    bt_lc = bt_lc[:,:,min_idx:max_idx+1]
    z_lc = z_lc[min_idx:max_idx+1]

    logger.info(f'CHUNKED BT')
    
    if noise_bt:
        bt_lc,_,_ = add_bt_noise(bt_lc,lc,z_lc)
        
    bt_lc = bt_lc.mean(axis=-1)

    logger.info(f'NOISE DONE')

    k_weights = np.ones_like(bt_lc)
    k_weights[bt_lc.shape[0]//2,bt_lc.shape[1]//2] = 0.
    res = get_power(
        bt_lc/bt_lc.mean() - 1,
        deltax2=x_lc/x_lc.mean() - 1,
        boxlength=x_lc.shape,
        bins=n_psbins,
        bin_ave=False,
        get_variance=False,
        log_bins=True,
        k_weights=k_weights,
    )
    
    return res, x_lc, bt_lc
