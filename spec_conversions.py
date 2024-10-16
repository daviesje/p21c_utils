import numpy as np
from astropy import units as U
from astropy import constants
from scipy import integrate
from numpy.random import normal

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(np.pi*2)) * np.exp(-(x-mu)**2/(2*sigma**2))

def abs_to_app(M,z,cosmo):
    d_ratio = (cosmo.luminosity_distance(z)/(10*U.pc)).to('').value
    return (M + 5*np.log10(d_ratio))

def sfr_to_Muv(sfr, fudge=None):
    # from Park 2018 https://arxiv.org/pdf/1809.08995.pdf
    kappa = 1.15e-28
    Luv = sfr * 3.1557e7 / kappa
    Muv = 51.64 - np.log10(Luv) / 0.4
    return Muv

def p_EW(
        Muv,
        beta=-2,
        mean=False,
        return_lum=True,
        high_prob_emit=False,
        EW_fixed=False
):
    """
    Function shall give sample from the distribution
    """

    def A(m):
        if high_prob_emit:
            return 0.95 + 0.05 * np.tanh(3 * (m+20.75))
        else:
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

    def W(m):
        return 31 + 12 * np.tanh(4 * (m + 20.25))

    if EW_fixed:
        C_const = 2.47 * 1e15 * U.Hz / 1216 / U.Angstrom * (
                1500 / 1216) ** (-(beta[0]) - 2)
        L_UV_mean = 10 ** (-0.4 * (Muv - 51.6))
        lum_alpha = W(Muv) * C_const.value * L_UV_mean

        return W(Muv), lum_alpha

    Ws = np.linspace(0, 500, 1000)

    if hasattr(Muv, '__len__') and not hasattr(beta, '__len__'):
        beta = beta * np.ones(len(Muv))

    C_const = 2.47 * 1e15 * U.Hz / 1216 / U.Angstrom * (1500 / 1216) ** (
                -(beta) - 2)
    L_UV_mean = 10 ** (-0.4 * (Muv - 51.6))

    if mean:
        if return_lum:
            return W(Muv) * A(Muv), W(Muv) * A(Muv) * C_const.value * L_UV_mean
        else:
            return W(Muv) * A(Muv)

    if hasattr(Muv, '__len__'):
        EWs = np.zeros((len(Muv)))
        if return_lum:
            lum_alpha = np.zeros((len(Muv)))
        for i, (muvi, beti) in enumerate(zip(Muv, beta)):
            if np.random.binomial(1, A(muvi)):
                EW_cumsum = integrate.cumtrapz(
                    1 / W(muvi) * np.exp(-Ws / W(muvi)), Ws)
                cumsum = EW_cumsum / EW_cumsum[-1]
                rn = np.random.uniform(size=1)
                EW_now = \
                np.interp(rn, np.concatenate((np.array([0.0]), cumsum)), Ws)[0]
            else:
                EW_now = 0.0
            EWs[i] = EW_now
            if return_lum:
                C_const = 2.47 * 1e15 * U.Hz / 1216 / U.Angstrom * (
                            1500 / 1216) ** (-(beti) - 2)
                L_UV_mean = 10 ** (-0.4 * (muvi - 51.6))
                lum_alpha[i] = EW_now * C_const.value * L_UV_mean
        if return_lum:
            return EWs, lum_alpha
        else:
            return EWs
    else:
        if np.random.binomial(1, A(Muv)):
            EW_cumsum = integrate.cumtrapz(1 / W(Muv) * np.exp(-Ws / W(Muv)),
                                           Ws)
            cumsum = EW_cumsum / EW_cumsum[-1]
            rn = np.random.uniform(size=1)
            EW_now = \
            np.interp(rn, np.concatenate((np.array([0.0]), cumsum)), Ws)[0]
            if return_lum:
                return EW_now, EW_now * C_const.value * L_UV_mean
            else:
                return EW_now
        else:
            if return_lum:
                return (0., 0.)
            else:
                return 0.

def get_js(
        muv=-18,
        z=7,
        n_iter=1,
        include_muv_unc=False,
        fwhm_true=False,
):
    """
    Function returns Lyman-alpha shape profiles

    :param muv: float,
        UV magnitude for which we're calculating the shape.
    :param z: float,
        redshift of interest.
    :param n_iter: integer,
        number of iterations of Lyman shape to get.
    :param include_muv_unc: boolean,
        whether to include the scatter in Muv.

    :return j_s: numpy.array of shape (N_iter, n_wav);
        array of profiles for a number of iterations and wavelengths.
    :return delta_vs : numpy.array of shape (N_iter);
        array of velocity offsets for a number of iterations.
    """
    n_wav = 100
    wave_em = np.linspace(1214, 1225., n_wav) * U.Angstrom

    wv_off = wave_to_dv(wave_em)
    #figure out how many iterations

    if hasattr(muv, '__len__'):
        tot_it_shape = (n_iter, *np.shape(muv))
    else:
        tot_it_shape = (n_iter)
    delta_vs = np.zeros(np.product(tot_it_shape))
    j_s = np.zeros((np.product(tot_it_shape), n_wav))

    if include_muv_unc and hasattr(muv, '__len__'):
        muv = np.array([np.random.normal(i, 0.1) for i in muv])
    elif include_muv_unc and not hasattr(muv, '__len__'):
        muv = np.random.normal(muv, 0.1)

    if hasattr(muv, '__len__'):
        delta_v_mean = np.array([delta_v_func(i,z) for i in muv.flatten()])
    else:
        delta_v_mean = delta_v_func(muv, z)

    for i in range(n_iter):
        if hasattr(muv, '__len__'):
            delta_vs[i] = 10**normal(delta_v_mean[i], 0.24)
        else:
            delta_vs[i] = 10**normal(delta_v_mean, 0.24)
        if fwhm_true:
            sigma=delta_vs[i] / 2 / np.sqrt(2 * np.log(2))
        else:
            sigma=delta_vs[i]
        j_s[i, :] = gaussian(wv_off.value, delta_vs[i], sigma)

    if  hasattr(muv, '__len__'):
        j_s.reshape((*tot_it_shape, n_wav))
        delta_vs.reshape(tot_it_shape)

    return j_s, delta_vs

def delta_v_func(
        muv,
        z=7
):
    """
    Function returns velocity offset as a function of UV magnitude

    :param muv: float,
        UV magnitude of the galaxy.
    :param z: float,
        redshift of the galaxy.

    :return delta_v: float
        velocity offset of a given galaxy.
    """
    if muv >= -20.0 - 0.26 * z:
        gamma = -0.3
    else:
        gamma = -0.7
    return 0.32 * gamma * (muv + 20.0 + 0.26 * z) + 2.34

def wave_to_dv(
        wave
):
    """
    Wavelength to velocity offset.
    :param wave: float;
        wavelength of interest.
    :return dv: float;
        velocity offset.
    """
    
    wave_Lya = 1215.67 * U.Angstrom
    return ((wave - wave_Lya)*constants.c/wave_Lya).to(U.km/U.s)

def Muv_to_Mlya(Muv, z, cosmo):
    n_wav = 100
    wave_em = np.linspace(1214, 1225., n_wav) * U.Angstrom
    _, Llya = p_EW(Muv, beta=-2, mean=False, return_lum=True, \
         high_prob_emit=False, EW_fixed=False)
    Flambda, dv = get_js(Muv)
    specific_intensity = (Llya*U.erg/U.s*Flambda.max()/U.Angstrom/integrate.trapz(
                                    Flambda,
                                    wave_em.value)*(
                                    (1215*U.Angstrom)**2/constants.c.cgs)).to(U.erg/U.Hz/U.s).value
    flux_lya = specific_intensity.max()/cosmo.luminosity_distance(z).to(U.cm).value**2
    mlya = -2.5*np.log10(flux_lya) - 48.6
    return mlya - cosmo.distmod(z).value
