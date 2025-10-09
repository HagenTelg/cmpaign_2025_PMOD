import pandas as pd
import xarray as xr

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