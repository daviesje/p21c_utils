import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import py21cmfast as p21c
import argparse

#a function which returns a parameter structure given a title, used for my test cases
def setup_params_from_title(title,hires=300,lores=100,boxlen=200,nthreads=1,minmass=None):
    #flag options
    stoc = "_SAMP" in title
    hf = "_SAMP" in title or "_DEXM" in title or "_FIX" in title
    fix = "_FIX" in title
    exf = "_EXP" in title
    rec = not "_NR" in title
    ts = not "_NOTS" in title
    cellr = "_CR" in title
    mini = "_MINI" in title
    
    if "_CONS-z" in title:
        cons = 'z-photoncons'
    elif "_CONS-a" in title:
        cons = 'alpha-photoncons'
    elif "_CONS-f" in title:
        cons = 'f-photoncons'
    else:
        cons = 'no-photoncons'

    zeta = not "_OLD" in title #turns off most of the other flags
    median = "_MEDIAN" in title
    compat = "_COMPAT" in title        
    
    fo = p21c.FlagOptions(HALO_STOCHASTICITY=stoc,USE_HALO_FIELD=hf,USE_MASS_DEPENDENT_ZETA=zeta
                        ,USE_TS_FLUCT=ts,INHOMO_RECO=rec,PHOTON_CONS_TYPE=cons,FIXED_HALO_GRIDS=fix
                        ,USE_EXP_FILTER=exf,USE_LYA_HEATING=True,CELL_RECOMB=cellr,USE_MINI_HALOS=mini
                        ,USE_CMB_HEATING=True,USE_UPPER_STELLAR_TURNOVER=not compat
                        ,HALO_SCALING_RELATIONS_MEDIAN=median
    )

    #user params
    if"_EPS" in title:
        hmf = "EPS"
    elif "_DELOS" in title:
        hmf = "DELOS"
    else:
        hmf = "ST"

    if "_FF" in title:
        intm = "GAMMA-APPROX"
    elif "_QAG" in title:
        intm = "GSL-QAG"
    else:
        intm = "GAUSS-LEGENDRE"

    if "_NUMLIM" in title:
        sampm = "NUMBER-LIMITED"
    elif "_PARTITION" in title:
        sampm = 'PARTITION'
    elif "_BINARY" in title:
        sampm = 'BINARY-SPLIT'
    else:
        sampm = "MASS-LIMITED"

    if minmass is None:
        minmass = 1e10 if '_LARGE' in title else 1e8
    #TODO:MAKE CLEANER
    if "_NT" in title:
        nthreads = int(title.split("_NT")[1][0])
        
    bufferfac = 5 if '_LARGE' in title else 2
    
    up = p21c.UserParams(USE_INTERPOLATION_TABLES=True,N_THREADS=nthreads,BOX_LEN=boxlen,DIM=hires,POWER_SPECTRUM='CLASS',
                                    HII_DIM=lores,HMF=hmf,USE_RELATIVE_VELOCITIES=True,
                                    MINIMIZE_MEMORY=False,INTEGRATION_METHOD_ATOMIC=intm,INTEGRATION_METHOD_MINI=intm,
                                    SAMPLER_MIN_MASS=minmass,MAXHALO_FACTOR=bufferfac,SAMPLE_METHOD=sampm)

    #astro params
    lx = 40 if compat else 40.5
    ap = p21c.AstroParams(L_X=lx)
    #AP matching SERRA
    if "_SERRA" in title:
        ap = ap.clone(F_STAR10=-0.8,ALPHA_STAR=0.1,SIGMA_STAR=0.25,t_STAR=0.15,
                                F_ESC10=-1.5,ALPHA_ESC=0.0)
    #AP matching ASTRID
    elif "_ASTRID" in title:
        ap = ap.clone(F_STAR10=-2.3,ALPHA_STAR=0.65,SIGMA_STAR=0.3,t_STAR=0.13,
                                F_ESC10=0.0,ALPHA_ESC=0.0)
    #No lognormal sigmas
    elif "_NOSIGMA" in title:
        ap = ap.clone(SIGMA_STAR=0.,SIGMA_SFR_LIM=0.,SIGMA_LX=0.)

    #cosmo params
    cp = p21c.CosmoParams()

    #globals
    filt = 1 if "_KS" in title else 0
    p21c.global_params.HII_FILTER = filt
    p21c.global_params.ZPRIME_STEP_FACTOR = 1.02
    p21c.global_params.DELTA_R_FACTOR = 1.05

    return cp,up,ap,fo
    
#one that works with the master branch
def setup_params_from_title_master(title,hires=300,lores=100,boxlen=200,nthreads=1):
    filt = 1 if "_KS" in title else 0
    cons = "_CONS" in title
    rec = not "_NR" in title
    rmax = 50 if "_RM" in title or rec else 15
    mini = "_MINI" in title
    ffcoll = "_FF" in title
    zeta = not "_OLD" in title #turns off most of the other flags
    halos = "_HALO" in title

    sub = not "_NOSUB" in title

    if"_ST" in title or halos:
        hmf = 1
    else:
        hmf = 0
    
    #TODO:MAKE CLEANER
    if "_NT" in title:
        nthreads = int(title.split("_NT")[1][0])
    
    p21c.global_params.ZPRIME_STEP_FACTOR = 1.02
    p21c.global_params.DELTA_R_FACTOR = 1.05
    p21c.global_params.HII_FILTER = filt
    p21c.global_params.USE_FAST_ATOMIC = ffcoll

    #AP matching SERRA
    if "_SERRA" in title:
        ap = p21c.AstroParams(F_STAR10=-0.8,ALPHA_STAR=-0.1,SIGMA_STAR=0.5,t_STAR=0.15,SIGMA_SFR=0.6)
    #AP matching ASTRID
    elif "_ASTRID" in title:
        ap = p21c.AstroParams(F_STAR10=-2.6,ALPHA_STAR=0.9,SIGMA_STAR=0.8,t_STAR=0.15,SIGMA_SFR=0.6)
    #Defaults
    else:
        ap = p21c.AstroParams()

    ap.update(R_BUBBLE_MAX=rmax)

    up = p21c.UserParams(USE_INTERPOLATION_TABLES=True,N_THREADS=nthreads,BOX_LEN=boxlen,DIM=hires,POWER_SPECTRUM=5
                                    ,HII_DIM=lores,HMF=hmf,USE_RELATIVE_VELOCITIES=True,FAST_FCOLL_TABLES=ffcoll)
                                    
    cp = p21c.CosmoParams()
    
    fo = p21c.FlagOptions(USE_MASS_DEPENDENT_ZETA=zeta,USE_TS_FLUCT=True,INHOMO_RECO=rec,PHOTON_CONS=cons
                        ,USE_LYA_HEATING=True,USE_MINI_HALOS=mini,USE_CMB_HEATING=True,USE_HALO_FIELD=halos)

    return cp,up,ap,fo

def parse_halotest_args():
    parser = argparse.ArgumentParser(description='test halo time correlations')
    parser.add_argument('--nthreads',type=int, default=1, help='number of OMP threads')
    parser.add_argument('--seed',type=int, help='random seed')
    parser.add_argument('--zstart',type=float,default=6,help='redshift start')
    parser.add_argument('--zstep',type=float,default=1.02,help='redshift step')
    parser.add_argument('--zstop',type=float,default=20,help='redshift')
    parser.add_argument('--hires',type=int,default=200,help='cell resolution')
    parser.add_argument('--lores',type=int,default=50,help='cell resolution')
    parser.add_argument('--boxlen',type=int,default=100,help='cell resolution')
    parser.add_argument('--minmass',type=float,default=5e7,help='min mass for sampler')
    parser.add_argument('--minfac',type=float,default=2,help='buffer factor for sampler')
    parser.add_argument('--suffix',type=str,default='',help='suffix for saving plots')
    parser.add_argument('--direc',type=str,default='/home/jdavies/plots/',help='directory to save plots')
    parser.add_argument('--method',type=int,default=0,help='sample methods')
    parser.add_argument('--hmf',type=int,default=1,help='HMF')
    parser.add_argument('--minlp',type=float,default=-16,help='min logprob in interptables')
    parser.add_argument('--npint',type=int,default=400,help='n prob bins in table')
    parser.add_argument('--ncint',type=int,default=200,help='n condition bins in table')

    args,remaining = parser.parse_known_args()

    return args,remaining