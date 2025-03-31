import numpy as np
import py21cmfast as p21c
import h5py
import pandas as pd
from astropy import units as U

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_closest_box(z_targets,class_type,lc):
    #get closest redshifts
    lc_z = lc.node_redshifts #redshifts at which we have coevals
    z_idx = np.argmin(np.fabs(lc_z[:,None] - z_targets[None,:]),axis=0)
    z_matches = lc_z[z_idx]

    hf_list = [class_type(cosmo_params=lc.cosmo_params,user_params=lc.user_params,
                        astro_params=lc.astro_params,flag_options=lc.flag_options,
                        random_seed=lc.random_seed,redshift=z) for z in z_matches]
    print(hf_list[0].filename)
    # print(hf_list[0].__repr__)

    return [hf.read(direc=p21c.config['direc']) for hf in hf_list], z_matches


def read_serra(cosmo,fname='data/serra_presentation_paper_data.csv'):
    df = pd.read_csv(fname,header=0,comment='#',sep=', ')
    df = df[['is_central','stellar mass','SFR','virial mass']]
    print(f'SERRA TOTAL {df.shape}')
    df_serra77 = df[df['is_central'] == 1.]
    print(f'SERRA CENTRAL {df.shape}')
    
    #SERRA ARRAYS z6 & z12
    log_mhalo_z_12 = np.array([       9.7141841052595,     10.178207332390324, 9.362217795768132,  9.743904383200798,  9.607911862717986,  10.427291579271209, 9.905685772923203, 9.388650170259856,  9.538839600321122,  9.193429366553604,  9.399913596867286,  9.876449081165967,  10.017972344765932, 9.867858523921551, 10.252638014804763, 9.927808044098699,   10.191945006497845, 9.165223133274202,  10.046413227974462, 9.759091552540095, 9.74745587618065,   10.166001344833276, 9.92273216736402,  9.762152261362939, 9.991201944719382,  9.80703390301268,   9.538935184542606, 9.857769263861982,  9.489509163077818,  9.87982705923636,  10.372540558467437, 10.08417013713104, 9.593597728239649,  10.300062919260254, 9.512210548288799,  9.77915250770349,  10.000441241844971, 9.287501400179583, 9.9910076713699,   9.539352291848354,  9.806795631101572, 9.879829244411516,  10.083992533860624, 10.372556255456182, 9.858183332948176, 9.490659488483159,  9.483134389568987, 9.592993824165537,  9.577275746858213,  10.021339054658414, 9.893191533098703, 9.436653897300948,  9.996414661897244, 9.828695411085533, 10.129221562193706,9.418745269524967, 9.131763814355569, 9.763442661729155, 9.781535651030982,  9.337045295828974,  9.700615491594256,  9.640923992650372, 9.659739239152877, 9.200520835191867])
    log_mstar_z_12 = np.array([       7.621902418518173,   8.555403952214006,  7.23182677051721,   7.843035682519098,  7.5649537808859755, 8.346857689669411,  7.507290563282561, 7.5476970761101,    7.52439478095069,   7.030689175477522,  7.158062159624806,  8.275607796694834,  8.390875955599666,  7.98247104067977,  8.59804184906585,   8.12107749804447,    8.50797184585973,   7.249969240693535,  8.374105601894383,  8.164052155844557, 7.804169020127476,  8.270431827661968,  8.282252298283511, 8.00853106504429,  7.573117742405499,  8.075558498502051,  7.880313789505161, 8.228109296199113,  7.529283445970964,  8.143664272634114, 8.63808252745102,   8.437388250869688, 7.832355739795015,  8.732559165039556,  7.799013729852771,  8.159405588976892, 7.555123071395756,  7.644001590230214, 7.54170249603707,  7.832605502420215,  8.08113135675085,  8.139584841337049,  8.46568385150165,   8.638392821911761,  8.222220261482466, 7.523799249791709,  7.524986716084793, 7.829400199548233,  7.791804061966596,  8.165763490472756,  7.971331877137484, 7.6716711532772806, 8.277534003325213, 8.026164374942644, 8.493879582055452, 7.342424214702917, 5.474588106981012, 6.04566113282534,  7.76127928962288,   7.4376637533962775, 8.115535873139878,  7.878495450713451, 7.79375944377143,  7.296805176949631])
    sfr_z_12 = np.array([ 0.37965513598538836, 6.356928158118572,  1.0556464895207542, 1.2773980135533143, 2.266030298284121,  7.10839890770288,   3.692755164868585, 2.2089356895061303, 5.1433384694420825, 0.8561027805143742, 2.5278888309528864, 3.2932760204800005, 9.576102471364488,  8.610086230502782, 11.71453766365982,  2.621948509136064,   20.551818708681267, 0.3414644510424838, 6.110890213062322,  4.288058853564202, 1.6703161315633646, 4.009608330781435,  3.898245240000935, 2.580961937781634, 1.9226082917839784, 3.1756579867827406, 2.217897888097551, 2.2517475585552997, 0.2533502662992665, 5.292957767330216, 11.584446694143535, 4.164932482112287, 3.2799542897753535, 11.035438578226024, 3.6181362482537316, 4.600883137701397, 2.4304305282002576, 2.495121791235473, 1.814987732227042, 2.3982688762237903, 3.016558253475504, 5.4873067756898894, 3.6654316997432006, 11.465327941579874, 2.225132263012885, 0.2575026373071005, 5.261483114154724, 3.3492271479521625, 3.3932346863287406, 2.388422523376479,  6.265867481004853, 2.316892804816883,  7.309223641889241, 4.024255148615896, 5.915963554619046, 3.635396758835745, 0.0,               0.0,               2.1107774994255046, 4.389959540197566,  2.9715531181673676, 2.946691694027182, 5.982117669105411, 2.3462949235994772])
    
    log_mhalo_z_6 = np.array([ 10.287713335454843, 11.838247835676341, 10.420641466809512, 10.159707372606885, 11.914289117702547, 9.800901637025381, 10.301591948307932, 10.621018752963092, 9.847887171183775, 10.413821174611872, 10.559994513027325, 10.023510920457138, 11.8439167145265, 10.596368105172019, 10.679503411060074, 10.59776515824973, 10.724618277738882, 10.692855819894932, 10.695620057559495, 12.115002378503121, 10.396692496334591, 10.80836615803218, 10.228481305983902, 9.796174519328908, 10.249890821091908, 10.842418274087525, 9.882787243504938, 10.3786478605974, 11.24728144496208, 10.544363400255882, 10.436993870800741, 10.244490010103078, 10.003820129114814, 10.645776877101138, 9.321082729568055, 11.483138033715688])
    log_mstar_z_6 = np.array([ 8.404366361763909, 10.184227699077573, 8.656214554135124, 8.589018354540862, 10.376787328794919, 8.241798263254964, 8.711504723387828, 8.598610248120083, 8.256544826492776, 8.43319013634767, 8.685690397977742, 8.225442822654712, 10.116354714992314, 8.878446715320283, 9.037115828296509, 9.225302385901058, 8.708024902107194, 9.139669985832557, 9.326388797585555, 10.361314122734456, 8.84916767774153, 9.187459936971152, 8.518273458217166, 8.350785266944452, 8.43017177297871, 9.139922770781176, 8.311781393309095, 8.750920251916849, 9.704189031298066, 9.110223325418618, 8.814422018535431, 8.591937087714824, 8.319835001673294, 9.10576679766068, 7.3059514589622205, 9.979222905917771])
    sfr_z_6 = np.array([ 2.420610392766052, 68.84106783654902, 6.637124041275937, 3.284063455678934, 548.439613203812, 1.116413347695131, 3.6540046816860308, 4.104624914345606, 0.9311227083234643, 2.6311991062850937, 2.553809739599423, 1.8090286828641406, 37.78957818322741, 3.499563089503395, 2.9220528202002067, 2.971879030055675, 8.065183210505046, 3.9932116090968313, 4.071447131476811, 126.07434977862003, 3.8532157243498855, 5.473865745185667, 0.9518536070126854, 0.4050596528832711, 3.444782930755906, 1.8983488647223175, 1.0753276550265385, 0.9818379843568458, 23.288147052580616, 2.359221757876334, 2.9113648808345935, 0.39299860680577275, 0.9392682591102315, 1.839287128207428, 0.0, 63.71537315850004])

    df_serra6 = pd.DataFrame.from_records(np.array([10**log_mhalo_z_6,10**log_mstar_z_6,sfr_z_6]).T,columns=['virial mass','stellar mass','SFR'])
    df_serra12 = pd.DataFrame.from_records(np.array([10**log_mhalo_z_12,10**log_mstar_z_12,sfr_z_12]).T,columns=['virial mass','stellar mass','SFR'])

    htime_yr6 = 1 / cosmo.H(6).to('yr-1').value
    df_serra6['SFR'] *= htime_yr6
    htime_yr77 = 1 / cosmo.H(7.7).to('yr-1').value
    df_serra77['SFR'] *= htime_yr77
    htime_yr12 = 1 / cosmo.H(12).to('yr-1').value
    df_serra12['SFR'] *= htime_yr12

    return df_serra6,df_serra77,df_serra12

def extract_run_fl(fl_str):
    return fl_str.split('_')[0]

def read_firstlight(z,htime_yr):
    fname = '~/data/FirstLight_SED_database.json'

    #The Firstlight database uses one object for each property for each galaxy (1 deep)
    #formatted in the title "OBJECT_TIME_PROPERTY"
    #There's probably a better way to extract this but it's a strange json organisation
    df = pd.read_json(fname,orient='index')

    #FL snapshots are spaced by a=0.001 so this SHOULD work for any z
    a = 1/(1+z)

    #filter out redshift 
    df = df.filter(like=f"_{a:.3f}_",axis=0)

    #get the 1-row sets of the data we need
    cols = ['Ms','Mvir','SFR']
    df_out = pd.DataFrame(columns=cols)
    for col in cols:
        subdf = df.filter(like=col,axis=0).rename(extract_run_fl,axis=0).rename({0:col},axis=1)
        df_out[col] = subdf

    #set SFR to units of Msun / hubble time
    df_out['SFR'] *= htime_yr
    return df_out

def read_astrid(cosmo,fname,z):
    logger.info(f"reading {fname}")
    df = pd.read_json(fname,orient='index')

    #set SFR to units of Msun / hubble time
    htime_yr = 1 / cosmo.H(z).to('yr-1').value
    df['StarFormationRate'] *= htime_yr

    return df

def parse_lc_list(lc_list):
    lightcones = []
    for lcf in lc_list:
        lc = read_lightcone(lcf)
        lightcones.append(lc)

    return lightcones

def read_cv_ignoreparams(fname):
    try:
        coeval = p21c.Coeval.read(fname)
    except ValueError:
        try:
            park, glbls = p21c.Coeval._read_inputs(fname)
            boxk = p21c.Coeval._read_particular(fname)
            #this sets globals to defaults, not consistent with the lightcone but allows us to load different versions
            coeval = p21c.Coeval(**park, **boxk)
            logger.warning(f"lightcone {fname} is likely from a different version and will have the wrong global parameters")
        except:
            logger.error(f"could not read lightcone {fname}")
            raise FileExistsError

    return coeval

def add_distances_to_lc(lcf):
    with h5py.File(lcf, "r+") as fl:
        if "distances" in fl:
            logger.info('file already has distances')
            return

        cosmo = p21c.CosmoParams(dict(fl["cosmo_params"].attrs)).cosmo
        boxlen = p21c.UserParams(dict(fl["user_params"].attrs)).BOX_LEN
        hiidim = p21c.UserParams(dict(fl["user_params"].attrs)).HII_DIM
        zdim = fl["lightcones"]["brightness_temp"].shape[-1]

        try:
            fl["distances"] = cosmo.comoving_distance(fl.attrs["redshift"]) + (np.arange(zdim,dtype=float)*boxlen/hiidim * U.Mpc)
        except:
            logger.info("couldn't add")
            pass
        logger.info(f"set lc dist {fl['distances']}")
    return

def add_mturn_to_lc(lcf):
    with h5py.File(lcf, "r+") as fl:
        if "log10_mturnovers" in fl or "log10_mturnovers_mini" in fl:
            logger.info('file already has turnovers')
            return

        try:
            fl["log10_mturnovers"] = np.zeros_like(fl["node_redshifts"])
            fl["log10_mturnovers_mini"] = np.zeros_like(fl["node_redshifts"])
        except:
            logger.info("couldn't add")
            pass
        logger.info(f"set lc mturn to zero")
    return

def inputs_from_database(filename):
    with h5py.File(filename) as hdf:
        inputs = p21c.InputParameters(
            node_redshifts=hdf['coeval_data'].attrs['node_redshifts'],
            user_params=dict(hdf['simulation_parameters']['user_params'].attrs),
            cosmo_params=dict(hdf['simulation_parameters']['cosmo_params'].attrs),
            astro_params=dict(hdf['simulation_parameters']['astro_params'].attrs),
            flag_options=dict(hdf['simulation_parameters']['flag_options'].attrs),
            random_seed=hdf.attrs['random_seed'],
        )
    return inputs

    
def lc_from_database(filename,quantities,global_quantities):
    name_map = {
        'density': 'density',
        'n_ion': 'ion_emissivity',
        'halo_mass' : 'M_halo',
        'halo_stars' : 'M_star',
        'halo_stars_mini' : 'M_star_mini',
        'halo_sfr' : 'SFR',
        'halo_sfr_mini' : 'SFR_mini',
        'xH_box': 'x_HI',
        'halo_xray': 'xray_emissivity',
        'brightness_temp': 'brightness_temp',
        'velocity_z': 'velocity_z',
    }
    inputs = inputs_from_database(filename)
    with h5py.File(filename) as hdf:
        inputs = p21c.InputParameters(
            node_redshifts=hdf['coeval_data'].attrs['node_redshifts'],
            user_params=dict(hdf['simulation_parameters']['user_params'].attrs),
            cosmo_params=dict(hdf['simulation_parameters']['cosmo_params'].attrs),
            astro_params=dict(hdf['simulation_parameters']['astro_params'].attrs),
            flag_options=dict(hdf['simulation_parameters']['flag_options'].attrs),
            random_seed=hdf.attrs['random_seed'],
        )
        lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
            min_redshift=inputs.node_redshifts[-1],
            max_redshift=inputs.node_redshifts[0],
            resolution=inputs.user_params.cell_size,
            cosmo=inputs.cosmo_params.cosmo,
            quantities=quantities,
        )

        lc = {
            quantity: hdf['lightcones'][name_map[quantity]][:]
            for quantity in quantities if quantity is not None
        }
        glbls = {
            quantity: hdf['coeval_data'][name_map[quantity]][:]
            for quantity in global_quantities if quantity is not None
        }

        lightcone = p21c.LightCone(
            lcn.lc_distances,
            inputs,
            lc,
            log10_mturnovers=hdf['coeval_data']['log10_mturnovers'],
            log10_mturnovers_mini=hdf['coeval_data']['log10_mturnovers_mini'],
            global_quantities=glbls,
        )

    return lightcone
    
def read_lightcone(fname,q=(None,),g_q=(None,),safe=False):
    if isinstance(fname,p21c.LightCone):
        lc = fname
    else:
        try:
            lc = p21c.LightCone.from_file(fname,safe=safe)
        except Exception as e:
            print(f"ERROR IN FROM_FILE {e}",flush=True)
            try:
                lc = p21c.LightCone.read(fname,safe=safe) #compatibility
            except Exception as e:
                print(f"ERROR IN READ {e}",flush=True)
                try:
                    lc = lc_from_database(fname,q,g_q)
                except Exception as e:
                    raise e
    
    return lc