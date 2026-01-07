import pandas as pd
import xarray as xr
import atmPy.radiation.retrievals.spectral_irradiance as atmspec
import atmPy.aerosols.physics.column_optical_properties as atmcop


def read_IT_CNR_ISAC_POM(p2f):

    def readone(p2f):
        df = pd.read_csv(p2f, skiprows=1)
        df.index = df.apply(lambda row: pd.to_datetime(f'{int(row.yyyy)}-{int(row.mm):02d}-{int(row.dd):02d} {int(row.HH_UTC):02d}:{int(row.MM):02d}:{int(row.SS):02d}'), axis = 1)
        
        aod = df.loc[:,[c for c in df if c[:3] == 'aod']]
        
        aod.columns = [int(c.replace('aod_', '')) for c in aod]
        
        aod.index.name = 'datetime'
        aod.columns.name = 'channel'
        
        ds = xr.Dataset()
        ds['aod'] = aod
        return ds
    if isinstance(p2f, list):
        dss = [readone(p) for p in p2f]
        ds = xr.concat(dss, dim='datetime')
        ds = ds.sortby('datetime')
        return ds
    else:
        ds = readone(p2f)
        return ds   

def read_ES_UV_POM(p2f):

    def readone(p2f):
        df = pd.read_csv(p2f, skiprows=1)
        df.index = df.apply(lambda row: pd.to_datetime(f'{int(row.yyyy)}-{int(row.mm):02d}-{int(row.dd):02d} {int(row.HH_UTC):02d}:{int(row.MM):02d}:{int(row.SS):02d}'), axis = 1)
        
        aod = df.loc[:,[c for c in df if c[:3] == 'aod']]
        
        aod.columns = [int(c.replace('aod_', '')) for c in aod]
        
        aod.index.name = 'datetime'
        aod.columns.name = 'channel'
        
        ds = xr.Dataset()
        ds['aod'] = aod
        return ds
    if isinstance(p2f, list):
        dss = [readone(p) for p in p2f]
        ds = xr.concat(dss, dim='datetime')
        ds = ds.sortby('datetime')
        return ds
    else:
        ds = readone(p2f)
        return ds   

def read_DE_PTB_SSIM(p2f):
    def readone(p2f):
        df = pd.read_csv(p2f)
        
        df.index = pd.DatetimeIndex(df['Time stamp'])
        
        aod = df.loc[:,[c for c in df if 'AOD' in c]]
        aod.columns = [int(c.split()[1]) for c in aod]
        
        aod.index.name = 'datetime'
        aod.columns.name = 'channel'
        
        ds = xr.Dataset()
        ds['aod'] = aod
        ds['aod'] = ds.aod.where(ds.aod != -1)
        return ds
    if isinstance(p2f, list):
        dss = [readone(p) for p in p2f]
        ds = xr.concat(dss, dim='datetime')
        ds = ds.sortby('datetime')
        return ds
    else:
        ds = readone(p2f)
        return ds

def read_SE_SMHI_PFR(p2f, no_wl = 4):
    def readone(p2f, no_wl = 4):
        df = pd.read_csv(p2f, skiprows=1, header=None, delimiter=r'\s+')
        
        with open(p2f) as rein:
            fline = rein.readline()    
            
        df.index = df.apply(lambda row: pd.to_datetime(f'{int(row[0]):04d}-{int(row[1]):02d}-{int(row[2]):02d} {int(row[3]):02d}:{int(row[4]):02d}:{int(row[5]):02d}'), axis = 1)
        df.index.name = 'datetime'
        df = df.drop(range(no_wl+1), axis=1)
            
        wls = fline.split()[2:]
        wlserr = ['d'+wl for wl in wls]
        wls = [float(wl) for wl in wls]
        cols = wls +['angstrom_exponent',  'angstrom_turbidity_coefficient'] + wlserr + ['pressure', 'ozone', 'no2' ]
        df.columns = cols

        #### convert to xr.Dataset
        ds = xr.Dataset()

        dft = df.iloc[:,:no_wl].copy()
        dft.columns.name = 'channel'
        ds['aod'] = dft
        
        start = no_wl+2
        end = start+no_wl
        dft = df.iloc[:,start:end].copy()
        dft.columns = [float(col.strip('d')) for col in dft.columns]
        dft.columns.name = 'channel'
        ds['aod_uncertainty'] = dft
        
        ds['angstrom_exponent'] = df.angstrom_exponent
        
        ds['angstrom_turbidity_coefficient'] = df.angstrom_turbidity_coefficient
        
        ds['pressure'] = df.pressure
        ds['ozone'] = df.ozone
        ds['no2'] = df.no2
        return ds
    if isinstance(p2f, list):
        dss = [readone(p, no_wl=no_wl) for p in p2f]
        ds = xr.concat(dss, dim='datetime')
        ds = ds.sortby('datetime')
        return ds
    else:
        ds = readone(p2f, no_wl=no_wl)
        return ds

def read_JP_JMA_POM(p2f, no_wl = 10):
    def readone(p2f, no_wl = 10):
        df = pd.read_csv(p2f, 
                        #skiprows=1, 
                        # header=None, 
                        delimiter=r'\s+')
        df
        
        df.index = df.apply(lambda row: pd.to_datetime(f'{p2f.name.split('.')[0].split('_')[-1]}') + pd.to_timedelta(row['%wl,'], 'm'), axis = 1)
        df.index = df.index - pd.to_timedelta(1, 'h')
        
        df.columns = [col.strip(',') for col in df.columns]
        
        df.index.name = 'datetime'
        df = df.drop('%wl', axis=1)
        
        
        
        #### convert to xr.Dataset
        ds = xr.Dataset()
        
        dft = df.iloc[:,:10].copy()
        dft.columns = [int(col) for col in dft.columns]
        dft.columns.name = 'channel'
        dft
        
        ds['aod'] = dft
        
        # start = no_wl+2
        # end = start+no_wl
        # dft = df.iloc[:,start:end].copy()
        # dft.columns = [float(col.strip('d')) for col in dft.columns]
        # dft.columns.name = 'channel'
        # ds['aod_uncertainty'] = dft
        
        # ds['angstrom_exponent'] = df.angstrom_exponent
        
        # ds['angstrom_turbidity_coefficient'] = df.angstrom_turbidity_coefficient
        
        # ds['pressure'] = df.pressure
        # ds['ozone'] = df.ozone
        # ds['no2'] = df.no2
        return ds
    if isinstance(p2f, list):
        dss = [readone(p, no_wl=no_wl) for p in p2f]
        ds = xr.concat(dss, dim='datetime')
        ds = ds.sortby('datetime')
        return ds
    else:
        ds = readone(p2f, no_wl=no_wl)
        return ds   

def get_langleys(ds, fnmet, lt,langley_airmass_limits = (2.5, 5), clean = False, test = False):
    """
    ds: dataset
    fnmet: path to metdata
    lt: prleliminary langleys for initail calibration for cloud screening, set to None for inital generation of langleys on a very clear day
    """
    out = {}
    ds = ds.copy()
    ds.attrs['site_longitude'] = 9.8458 
    ds.attrs['site_latitude'] = 46.8143 
    ds.attrs['site_elevation'] = 1590
    
    # calibrate with preliminary calibration (the last ones)
    # the following is solely to create a cloudmask tha can be applied before performing the langley calibration
    gdd = atmspec.CombinedGlobalDiffuseDirect(ds.copy())
    gdd.path2solar_spectrum = '/Users/htelg/fundgrube/reference_data/solar/spectrum/solar_spectral_irradiance_e490_00a_amo.nc'
    gdd.dataset['channel_wavelength'] = gdd.dataset.channel_wavelength.astype(float) #error in processing upstream, fix it in future versions!
    if not isinstance(lt, type(None)):
        gddc = gdd.apply_calibration_langley(lt)
    
        dnic = gddc.direct_normal_irradiation
        out['dnic'] = dnic
        dnic.dataset = dnic.dataset.where(dnic.dataset.channel < 1000, drop = True)
        dnic.met_data = fnmet
        dnic.ozone_data = 300
        dnic.ozone_absorption_spectrum = '/Users/htelg/fundgrube/reference_data/materials/ozon/ozone.coefs'
        # dnic.aod.plot.line(x = 'datetime')
        # dnic.aod.plot.line(x = 'datetime')
        # plt.ylim(0,0.05)
        
        # get cloudmask
        aodi = atmcop.AOD_AOT(dnic.aod)
        out['aodi'] = aodi
        cloudmask = aodi.cloudmask.cloudmask_michalsky.drop_vars('channel')
        out['cloudmask'] = cloudmask
    
    #### Make the langleys
    si = atmspec.CombinedGlobalDiffuseDirect(ds)
    sir = si.direct_normal_irradiation
    sir.settings_langley_airmass_limits = langley_airmass_limits
    out['sir'] = sir
    # apply cloudmask
    if not isinstance(lt, type(None)):
        sir.dataset = sir.dataset.where(cloudmask == 0)

    if test:
        return out
    if clean:
        lang = sir.langley_am
        
        # lang.plot(wavelength=None, show_pre_clean=False)
        # clean the langley
        lc = lang.clean(threshold=2)
        langc_am = lc['langley']
        
        lang = sir.langley_pm
        lc = lang.clean(threshold=2)
        langc_pm = lc['langley']
        
        return langc_am, langc_pm
    else:
        return sir.langley_am, sir.langley_pm