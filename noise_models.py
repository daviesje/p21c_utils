'''
Noise / telescope models for analysing 21cmfast output

Many of these are David Prelogovic's taken from
https://github.com/dprelogo/21cmRNN/blob/main/rnn21cm/database.py
and Slack messages from David
'''
import numpy as np
import jax.numpy as jnp
from scipy.integrate import quadrature
from scipy.ndimage import fourier_gaussian
import tools21cm as t2c
from tqdm import tqdm
from astropy import units as U, constants as c
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_uv_coverage(Box_uv, uv_bool):
    """Apply UV coverage to the data.
    Args:
        Box_uv: data box in Fourier space
        uv_bool: mask of measured baselines
    Returns:
        Box_uv
    """
    Box_uv = Box_uv * uv_bool
    return Box_uv


def compute_uv_coverage(redshifts, ncells=200, boxsize=300):
    """Computing UV coverage box for SKA antenna configuration.
    Args:
        redshifts: list of redshifts for which the UV coverage is computed.
        ncells: lsize of a grid in UV space (in pixels)
        boxsize: size of the simulation (in Mpc)
    Returns:
        uv: UV coverage box
    """
    uv = np.empty((ncells, ncells, len(redshifts)))

    for i in range(len(redshifts)):
        print(f'{i} of {len(redshifts)}', end=" ", flush=True)
        uv[..., i], _ = t2c.noise_model.get_uv_map(
            ncells=ncells, z=redshifts[i], boxsize=boxsize
        )

    return uv


def noise(seed, redshifts, uv, ncells=200, boxsize=300.0, obs_time=1000, N_ant=512):
    """Computing telescope thermal noise.
    Args:
        seed: noise seed
        redshifts: list of redshifts for each slice of UV
        uv: UV coveragebox
        ncells: size of a box in real/UV space (in pixels)
        boxsize: size of the simulation (in Mpc)
        obs_time: total observation time (in hours)
        N_ant: number of antennas in the configuration
    Returns:
        finalBox: noise in UV space
    """
    redshifts = np.append(
        redshifts, 2 * redshifts[-1] - redshifts[-2]
    )  # appending the last difference
    finalBox = np.empty(uv.shape, dtype=np.complex64)
    for i in range(uv.shape[-1]):
        depth_mhz = t2c.cosmology.z_to_nu(redshifts[i]) - t2c.cosmology.z_to_nu(
            redshifts[i + 1]
        )
        noise = t2c.noise_model.noise_map(
            ncells=ncells,
            z=redshifts[i],
            depth_mhz=depth_mhz,
            obs_time=obs_time,
            boxsize=boxsize,
            uv_map=uv[..., i],
            N_ant=N_ant,
        )
        noise = t2c.telescope_functions.jansky_2_kelvin(
            noise, redshifts[i], boxsize=boxsize, ncells=ncells
        ).astype(np.complex64) #this is actually muJy to mK???
        finalBox[..., i] = noise

    return finalBox

def noise_smart(seed, uv, sigma):
    RState = np.random.RandomState(seed=seed)
    noise = RState.normal(loc=0.0, scale=1.0, size=(2,) + uv.shape) * sigma[None,None,None,:] #cmplx,u,v,nu
    noise = noise[0] + 1.j*noise[1]
    noise /= np.sqrt(uv)
    return noise

def save_noise_arrays(lc, obs_time=1000., int_time=10., uvfile=None):
    ncells = lc.user_params.HII_DIM
    boxsize = lc.user_params.BOX_LEN
    z_lc = lc.lightcone_redshifts

    if uvfile is None:
        uv_map = compute_uv_coverage(z_lc, ncells=ncells, boxsize=boxsize)
        np.save(f'./data/uv_l{boxsize:d}_c{ncells:d}',uv_map)
    else:
        uv_map = np.load(uvfile)

    rms_noi = np.zeros_like(z_lc)
    rms_example = np.load('./data/sigma.npy')
    print(uv_map.shape,z_lc.shape,rms_example.shape)
    for i in range(uv_map.shape[-1] - 1):
        depth_mhz = t2c.cosmology.z_to_nu(z_lc[i]) - t2c.cosmology.z_to_nu(
            z_lc[i + 1]
        )
        _, rms_noi[i] = t2c.telescope_functions.kanan_noise_image_ska(z_lc[i],
                                         uv_map[...,i], depth_mhz, obs_time, int_time, verbose=False)
                                         
        #convert to K
        rms_noi[i] = t2c.telescope_functions.jansky_2_kelvin(rms_noi[i], z=z_lc[i], ncells=ncells,
                            boxsize=boxsize).astype(np.complex64) * np.sqrt(int_time/3600./obs_time)
        if i % 10 == 0:
            print(f'Noise at z={z_lc[i]} == {rms_noi[i]} mK {rms_example[i]} example ratio {rms_noi[i]/rms_example[i]}')

    np.save(f'./data/rms_l{boxsize:d}_c{ncells:d}',rms_noi)

def wedge_removal_box(
    OMm,
    redshifts,
    cell_size,
    t_box,
    blackman=True,
):
    box_shape = t_box.shape
    print(box_shape)
    chunk_length = box_shape[-1]
    HII_DIM = box_shape[0]

    def one_over_E(z, OMm):
        return 1 / np.sqrt(OMm * (1.0 + z) ** 3 + (1 - OMm))

    def multiplicative_factor(z, OMm):
        return (
            1
            / one_over_E(z, OMm)
            / (1 + z)
            * quadrature(lambda x: one_over_E(x, OMm), 0, z)[0]
        )

    redshifts = np.array(redshifts).astype(np.float32)
    MF = multiplicative_factor(redshifts[-1], OMm)

    k = 2 * np.pi * np.fft.fftfreq(HII_DIM, d=cell_size)
    k_parallel = 2 * np.pi * np.fft.fftfreq(chunk_length, d=cell_size)
    delta_k = k_parallel[1] - k_parallel[0]
    k_cube = np.meshgrid(k, k, k_parallel)

    bm = np.abs(np.fft.fft(np.blackman(chunk_length))) ** 2
    buffer = delta_k * (np.where(bm / np.amax(bm) <= 1e-10)[0][0] - 1)
    BM = np.blackman(chunk_length)[np.newaxis, np.newaxis, :]

    W = k_cube[2] / (np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2) * MF + buffer)
    w = np.logical_or(W < -1.0, W > 1.0)
    if blackman == True:
        t_box = t_box * BM
    return np.real(np.fft.ifft2(t_box, axes=(0,-1)) * w)

def wedge_removal_old(
    OMm,
    redshifts,
    HII_DIM,
    cell_size,
    Box_uv,
    chunk_length=501,
    blackman=True,
    padding="zeros",
):
    """Computing horizon wedge removal. Implements "sliding" procedure
    of removing the wedge for every redshift separately.

    Args:
        OMm: Omega matter
        redshifts: list of redshifts in a lightcone
        HII_DIM: size of the HII simulation box (see `21cmFASTv3`)
        cell_size: size of a cell in Mpc
        Box_uv: box in UV space on which wedge removal is to be computed
        chunk_length: length of a sliding chunk (in number of z-slices)
        blackman: either to use Blackman-Harris taper or not
        padding: either "zeros" or "mirrored", defining the type of padding on the left and right

    Returns:
        Box_final: wedge-removed box in real space
    """
    
    if padding not in ("zeros", "mirrored"):
        raise ValueError(f"padding can be either zeros or mirrored, but is {padding}.")

    def one_over_E(z, OMm):
        return 1 / np.sqrt(OMm * (1.0 + z) ** 3 + (1 - OMm))

    def multiplicative_factor(z, OMm):
        return (
            1
            / one_over_E(z, OMm)
            / (1 + z)
            * quadrature(lambda x: one_over_E(x, OMm), 0, z)[0]
        )

    MF = jnp.array([multiplicative_factor(z, OMm) for z in redshifts]).astype(
        np.float32
    )
    print("_____SLOPE_____")
    print(MF)
    redshifts = jnp.array(redshifts).astype(np.float32)

    k = jnp.fft.fftfreq(HII_DIM, d=cell_size)
    k_parallel = jnp.fft.fftfreq(chunk_length, d=cell_size)
    delta_k = k_parallel[1] - k_parallel[0]
    k_cube = jnp.meshgrid(k, k, k_parallel)

    bm = jnp.abs(jnp.fft.fft(jnp.blackman(chunk_length))) ** 2
    buffer = delta_k * (jnp.where(bm / jnp.amax(bm) <= 1e-10)[0][0] - 1)
    BM = jnp.blackman(chunk_length)[jnp.newaxis, jnp.newaxis, :]

    box_shape = Box_uv.shape
    Box_final = np.empty(box_shape, dtype=np.float32)
    Box_uv = jnp.array(Box_uv, dtype=jnp.complex64)
    if padding == "zeros":
        empty_box = jnp.zeros(k_cube[0].shape)
        Box_uv = jnp.concatenate((empty_box, Box_uv, empty_box), axis=2)
    else:
        Box_uv = jnp.concatenate(
            (
                Box_uv[:chunk_length][::-1], 
                Box_uv, 
                Box_uv[-chunk_length:][::-1]
            ),
            axis=2,
        )

    for i in tqdm(range(chunk_length, box_shape[-1] + chunk_length)):
        t_box = Box_uv[..., i - chunk_length // 2 : i + chunk_length // 2 + 1]
        W = k_cube[2] / (
            jnp.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2)
            * MF[min(i - chunk_length // 2 - 1, box_shape[-1] - 1)]
            + buffer
        )
        w = jnp.logical_or(W < -1.0, W > 1.0)
        if blackman == True:
            t_box = t_box * BM
        Box_final[..., i - chunk_length] = jnp.real(
            jnp.fft.ifftn(jnp.fft.fft(t_box, axis=-1) * w)
        )[
            ..., chunk_length // 2
        ]  # taking only middle slice in redshift

    return Box_final.astype(np.float32)

def wedge_removal(
    cosmo,
    redshifts,
    HII_DIM,
    cell_size,
    Box_uv,
    chunk_length=501,
    blackman=True,
    padding="zeros",
):
    """Computing horizon wedge removal. Implements "sliding" procedure
    of removing the wedge for every redshift separately.
    Args:
        OMm: Omega matter
        redshifts: list of redshifts in a lightcone
        HII_DIM: size of the HII simulation box (see `21cmFASTv3`)
        cell_size: size of a cell in Mpc
        Box_uv: box in UV space on which wedge removal is to be computed
        chunk_length: length of a sliding chunk (in number of z-slices)
        blackman: either to use Blackman-Harris taper or not
    Returns:
        Box_final: wedge-removed box in real space
    """

    #NOTE: this slope is slightly different but by <1%
    slope = (cosmo.comoving_distance(redshifts) * cosmo.H(redshifts)/(c.c * (1+redshifts))).to('').value
    print("_____SLOPE_____")
    print(slope)

    #QUESTION 1: If we ignore the 2pi here, buffer is set to be much smaller, is this an issue?
    k_perpendicular = 2 * jnp.pi * jnp.fft.fftfreq(HII_DIM, d=cell_size)
    k_parallel = 2 * jnp.pi * jnp.fft.fftfreq(chunk_length, d=cell_size)
    
    #IN DAVID'S FUCNTION IT IS EQUIVALENT TO:
    #k_perpendicular = jnp.fft.fftfreq(HII_DIM, d=cell_size)
    #k_parallel = jnp.fft.fftfreq(chunk_length, d=cell_size)
    
    delta_k = jnp.fft.fftshift(k_parallel)[1] - jnp.fft.fftshift(k_parallel)[0]
    k_cube = jnp.meshgrid(k_perpendicular, k_perpendicular, k_parallel,indexing='ij')
    
    bm = jnp.abs(jnp.fft.fft(jnp.blackman(chunk_length))) ** 2 #power in each k-bin of the BH taper
    #Buffer is the first k at which the taper funtion is less than 1e-10 of the maximum mode
    buffer = delta_k * (jnp.where(bm / jnp.amax(bm) <= 1e-10)[0][0] - 1)
    BM = jnp.blackman(chunk_length)[jnp.newaxis, jnp.newaxis, :] #blackman taper
    
    print(f'dk {delta_k} buffer {buffer}')

    box_shape = Box_uv.shape
    Box_final = np.empty(box_shape, dtype=jnp.float32)
    zero_padding = jnp.zeros_like(Box_uv[:,:,:chunk_length])
    reflect_left = Box_uv[:,:,chunk_length::-1]
    reflect_right = Box_uv[:,:,-chunk_length::-1]
    
    if padding == "zeros":
        Box_uv = jnp.concatenate(
            (zero_padding, jnp.array(Box_uv, dtype=jnp.float32), zero_padding), axis=2
        )
    elif padding == "reflect":
        Box_uv = jnp.concatenate(
            (reflect_left, jnp.array(Box_uv, dtype=jnp.float32), reflect_right), axis=2
        )
    else:
        raise ValueError("wrong padding string")
        
    print(Box_uv.shape, chunk_length, box_shape[-1] + chunk_length + chunk_length // 2 + 1)

    k_perp = jnp.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2)
    k_pll = k_cube[2]
    
    for i in tqdm(range(chunk_length, box_shape[-1] + chunk_length)):
        t_box = Box_uv[..., i - chunk_length // 2 : i + chunk_length // 2 + 1]
        
        #QUESTION 2: why this index, surely we want the same as the initial lightcone?
        #IN DAVID'S
        slope_idx = min(i - chunk_length // 2 - 1, box_shape[-1] - 1)
        # slope_idx = i - chunk_length
        w = jnp.fabs(k_pll / (k_perp*slope[slope_idx] + buffer)) > 1.
        if blackman == True:
            t_box = t_box * BM

        #For the real box, FT the z-axis of the chunk, ifft the whole thing and
        #   assign the central slice to the correct redshift
        Box_final[..., i - chunk_length] = jnp.real(
            jnp.fft.ifftn(jnp.fft.fft(t_box, axis=-1) * w)
        )[
            ..., chunk_length // 2
        ]  # taking only middle slice in redshift

    return Box_final.astype(jnp.float32)

def BoxCar3D(data, filter=(4, 4, 4)):
    """Computing BoxCar filter on the input data.
    Args:
        data: data to filter
        filter: filter shape
    Returns:
        filtered data
    """
    if len(data.shape) != 3:
        raise AttributeError("data has to be 3D")
    if len(filter) != 3:
        raise AttributeError("filter has to be 3D")
    s = data.shape
    Nx, Ny, Nz = filter

    return np.einsum(
        "ijklmn->ikm",
        data[: s[0] // Nx * Nx, : s[1] // Ny * Ny, : s[2] // Nz * Nz].reshape(
            (s[0] // Nx, Nx, s[1] // Ny, Ny, s[2] // Nz, Nz)
        ),
    ) / (Nx * Ny * Nz)


def add_bt_noise(bt,lc,z_lc,wedge=True,thermal=True,uvcov=True,seed=1234):
    ncells = lc.user_params.HII_DIM
    boxlen = lc.user_params.BOX_LEN
    cellsize = boxlen / ncells
    print(f'adding noise to BT LC of shape {bt.shape}')
    min_uv = 15
    max_base = 2 * U.km
    # min_uv = 1
    # max_base = 100 * U.km

    #Remove mean of each frequency slice
    bt = bt - bt.mean(axis=(0,1))[None,None,:]

    bt_uv = np.fft.fft2(bt,axes=(0,1))
    
    noise_arr = np.zeros_like(bt_uv)
    uv_sel = np.ones_like(bt_uv)

    k_uv = None
    if uvcov:
        if ncells == 200 and boxlen == 300 and np.isclose(z_lc.min(),5.,atol=1e-3):
            uv = np.load('./data/uv.npy')
            uv_sel = np.load('./data/uv_bool_15.npy')
            logger.info(f"loaded uv_sel {uv_sel.shape}")
        elif ncells == 100 and boxlen == 200 and np.isclose(z_lc.min(),6.):
            uv = np.load('./data/uv_l200_c100.npy')
        else:
            raise ValueError(f"Don't have that UV file shape BOX_LEN == {boxlen} HII_DIM == {ncells} z={z_lc.min()}")
        print(f'USING UV FILE OF SHAPE {uv.shape}')
        k_uv = np.array(np.meshgrid(np.fft.fftfreq(uv.shape[0]),
                        np.fft.fftfreq(uv.shape[1]),indexing='ij')) #Mpc-1 [2,x,y]
        print(k_uv.shape)
        k_uv = k_uv * 2*np.pi / cellsize
        #NOTE: I'm assuming here that the redshifts line up (i.e same minimum and step in redshift, but with possibly different maxima)
        #   This doesn't work if UV has fewer entries than the lightcone
        uv = uv[...,:bt.shape[-1]]

        if np.all(uv_sel == 1):
            lambda_lc = 21 * U.cm * (1+z_lc)
            ang_limit = (lambda_lc / max_base).to('').value
            d_a_lc = lc.cosmo_params.cosmo.angular_diameter_distance(z_lc)
            k_lim = (2*np.pi / (d_a_lc * (1+z_lc) * ang_limit)).to('Mpc-1').value

            uv_sel = (uv>=min_uv)*(np.linalg.norm(k_uv,axis=0)[:,:,None] < k_lim[None,None,:])
            print(f'K < {max_base} || z {z_lc[db_idx]} {k_lim[db_idx]}')
        else:
            uv_sel = uv_sel[...,:bt.shape[-1]]

        if thermal:
            sigma = None
            # if ncells == 200 and boxlen == 300 and np.isclose(z_lc.min(),5.):
            #     sigma = np.load('./data/sigma_redo.npy')
            # elif ncells == 100 and boxlen == 200 and np.isclose(z_lc.min(),6.):
            #     sigma = np.load('./data/sigma_l200_c100.npy')

            if sigma is not None:
                sigma = sigma[...,:bt.shape[-1]]
                print(f'USING SIGMA FILE OF SHAPE {sigma.shape}')
                noise_arr = noise_smart(seed=seed,uv=uv,sigma=sigma)
            else:
                noise_arr = noise(seed, z_lc, uv, ncells=ncells, boxsize=boxlen)
                noise_arr = np.fft.fft2(noise_arr,axes=(0,1))

            ##debug###
            db_redshifts = np.array([6.1,12,24])
            db_redshifts = db_redshifts[(db_redshifts<=z_lc.max()) & (db_redshifts>=z_lc.min())]
            db_idx = np.argmin(np.fabs(db_redshifts[None,:]-z_lc[:,None]),axis=0)
            ###\debug###
            
            print(f"UVCOV: removed {(1 - uv_sel).sum()} of {uv.size} modes {(1 - uv_sel).sum()/uv.size*100} %")
            print(f"THERMAL: Mean nonzero noise at z={z_lc[db_idx]}"
                    +f"= {np.absolute((noise_arr*uv_sel)[...,db_idx]).sum(axis=(0,1))/np.count_nonzero((noise_arr*uv_sel)[...,db_idx],axis=(0,1))} mK")

    logger.info(f"BT shape {bt_uv.shape} MASK {uv_sel.shape}")
    bt_uv = bt_uv + noise_arr
    bt_uv = bt_uv*uv_sel

    if wedge:
            bt_noised_ft = wedge_removal_old(lc.cosmo_params.OMm,z_lc,ncells,cellsize,bt_uv,chunk_length=501)
            # bt_noised_ft = wedge_removal(lc.cosmo_params.cosmo,z_lc,ncells,cellsize,bt_uv,chunk_length=701)
    else:
        bt_noised_ft = np.real(np.fft.ifft2(bt_uv,axes=(0,1)))

    return bt_noised_ft, np.real(np.fft.ifft2(noise_arr*uv_sel,axes=(0,1))), k_uv

def add_cii_noise(B_cii,nu_lc,lc_obj,noisefile=None,seed=1234,hours=1):
    NU_CII = 1900.54 * U.GHz
    RState = np.random.RandomState(seed=seed)

    with fits.open(noisefile) as f_in:
        data = f_in[1].data
        nu_sens = data['nu'] * U.GHz #GHz
        bw_sens = data['bandwidth'] * U.GHz #GHz
        octiles = (np.arange(1,8) * 12.5)
        val_sens = [data['RMS_noise [octile {0:.1f}]'.format(octile)] for octile in octiles] #RMS for white noise
        fwhm_sens = data['FWHM_beam'] * U.deg #degrees
        z_sens = NU_CII/nu_sens - 1

    #angular size of cell in bands
    d_a_sens = lc_obj.cosmo_params.cosmo.angular_diameter_distance(z_sens)
    ang_size_sens = ((lc_obj.user_params.BOX_LEN/lc_obj.user_params.HII_DIM) / (1+z_sens) * U.Mpc / d_a_sens) * U.rad

    B_conv = np.zeros((B_cii.shape[0],B_cii.shape[1],nu_sens.size))
    noise = np.zeros((B_cii.shape[0],B_cii.shape[1],nu_sens.size))
    B_out = np.zeros_like(B_cii)
    noise_out = np.zeros_like(B_cii)
    print(f'nu range in LC {nu_lc.min()},{nu_lc.max()}')
    print(f'noisefile covers redshifts {z_sens.min()},{z_sens.max()}')
    beam_sigma = ((fwhm_sens*0.5) / ang_size_sens).to('').value #half FWHM as sigma, cell units
    for i in range(nu_sens.size):
        bw = bw_sens[i]
        nu_min = (nu_sens[i] - bw/2)
        nu_max = (nu_sens[i] + bw/2)

        #I'm assuming a strict binning here, which works best for bw > cell
        #TODO: an interpolation + boxcar filter may be more realistic since that's how we treat the
        #   simulation fields
        sel = (nu_lc > nu_min) & (nu_lc < nu_max)
        if np.all(sel==False):
            continue
        B_real = B_cii[...,sel].mean(axis=-1)

        octile_sel = 3 #median
        # octile_sel = np.random.randint(7,shape=B_real.shape) #random octile per mode?
        
        #fft
        B_fourier = np.fft.fft2(B_real,axes=(0,1))

        #add thermal noise (shamelessly stolen from David)
        #assume independent per frequency channel and fourier mode
        noise_fourier = RState.normal(loc=0.0, scale=1.0, size=(2,) + B_fourier.shape) * val_sens[octile_sel][i] / np.sqrt(hours)
        noise_fourier = (noise_fourier[0] + 1.j*noise_fourier[1])

        B_fourier = B_fourier + noise_fourier

        #filter by beam fwhm, assume gaussian
        #TODO:ask someone if the noise should be applied before or after beam filtering
        B_fourier = fourier_gaussian(B_fourier,sigma=beam_sigma[i])
        B_conv[...,i] = np.real(np.fft.ifft2(B_fourier,axes=(0,1)))
        noise[...,i] = np.real(np.fft.ifft2(noise_fourier,axes=(0,1)))

    #rebin to grid scale so we can do crosses and stufff
    #TODO: figure out the best place to do this interpolation, in k-space perhaps?
    # interpB = interp1d(nu_sens.to('GHz').value,B_conv,bounds_error=False)
    # B_cii[...] = interpB(nu_lc[0,0,:].to('GHz').value)

    for x in range(B_cii.shape[0]):
        for y in range(B_cii.shape[1]):
            B_out[x,y,:] = np.interp(nu_lc,nu_sens,B_conv[x,y,:])
            noise_out[x,y,:] = np.interp(nu_lc,nu_sens,noise[x,y,:])

    print(f'Before Noise {np.nanmin(B_cii)} {np.nanmax(B_cii)} {np.nanmean(B_cii)}')
    print(f'1st interp {np.nanmin(B_real)} {np.nanmax(B_real)} {np.nanmean(B_real)}')
    print(f'After Noise {np.nanmin(B_conv)} {np.nanmax(B_conv)} {np.nanmean(B_conv)}')
    print(f'2nd Interp {np.nanmin(B_out)} {np.nanmax(B_out)} {np.nanmean(B_out)}')
    
    # fig,ax = plt.subplots(nrows=1,ncols=2)
    # ax[0].semilogy(nu_sens,val_sens)
    # ax[0].set_xlabel('nu')
    # ax[0].set_ylabel('noise sigma Jy')
    # ax[1].plot(nu_sens,beam_sigma)
    # ax[1].set_xlabel('nu')
    # ax[1].set_ylabel('beam FWHM cells')
    # ax[0].set_ylim([1e-4,1e-1])
    # fig.savefig('./test_atlast.png')

    noise_info = { "beam_fwhm_cells" : beam_sigma, "redshift" : z_sens, "bandwidth" : bw_sens}

    # print(f'nu {nu_sens.min():.2e} to {nu_sens.max():.2e}: sens mean {val_sens.mean()} fhwm mean {beam_sigma.mean()}')
    return B_out, noise_out, noise_info
