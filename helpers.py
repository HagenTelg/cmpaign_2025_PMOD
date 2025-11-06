import pandas as pd
import xarray as xr
import atmPy.radiation.retrievals.spectral_irradiance as atmspec
import atmPy.aerosols.physics.column_optical_properties as atmcop


def read_SE_SMHI_PFR(p2f, no_wl = 4):
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


def read_JP_JMA_POM(p2f, no_wl = 10):
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

def get_langleys(ds, fnmet, lt,langley_airmass_limits = (2.5, 5), test = False):
    """
    ds: dataset
    fnmet: path to metdata
    lt: prleliminary langleys for initail calibration for cloud screening
    """
    ds = ds.copy()
    ds.attrs['site_longitude'] += 360 
    
    # calibrate with preliminary calibration (the last ones)
    gdd = atmspec.CombinedGlobalDiffuseDirect(ds.copy())
    gdd.dataset['channel_wavelength'] = gdd.dataset.channel_wavelength.astype(float) #error in processing upstream, fix it in future versions!
    
    gddc = gdd.apply_calibration_langley(lt)
    
    dnic = gddc.direct_normal_irradiation
    dnic.raw_data = dnic.raw_data.where(dnic.raw_data.channel < 1000, drop = True)
    dnic.met_data = fnmet
    dnic.ozone_data = 300
    # dnic.aod.plot.line(x = 'datetime')
    # dnic.aod.plot.line(x = 'datetime')
    # plt.ylim(0,0.05)
    
    # get cloudmask
    aodi = atmcop.AOD_AOT(dnic.aod)
    cloudmask = aodi.cloudmask.cloudmask_michalsky.drop_vars('channel')
    
    #### Make the langleys
    si = atmspec.CombinedGlobalDiffuseDirect(ds)
    sir = si.direct_normal_irradiation
    sir.settings_langley_airmass_limits = langley_airmass_limits
    # apply cloudmask
    sir.raw_data = sir.raw_data.where(cloudmask == 0)

    clean = False
    if test:
        out = dict(sir = sir, cloudmask = cloudmask, aodi = aodi, dnic = dnic)
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