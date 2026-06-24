import pandas as pd
import xarray as xr
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import atmPy.radiation.retrievals.spectral_irradiance as atmspec
import atmPy.aerosols.physics.column_optical_properties as atmcop
import atmPy.radiation.retrievals.langley_calibration as atmlc
import copy
import pathlib as pl
import scipy
# def plot_pwvcal(pwvcal, color, ax = None):
#     if ax is None:
#         f,a = plt.subplots()
#     else:
#         a = ax
#         f = ax.get_figure()
#     pwvcal.pwvcal_pre_clean.plot(ax = a, label='am', textpos=(0.6,0.85), data_kwargs={'lw':4, 'alpha':0.5, 'color': color}, fit_kwargs={'alpha':0})
#     pwvcal.plot(ax = a, label='am', textpos=(0.6,0.85), 
#                    data_kwargs={'color':color, 'ls':'', 'marker':'.'}
#                   )
#     return f,a

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

def get_langleys(ds, fnmet, lt,langley_airmass_limits = (2.5, 5), clean = False, test = False, cal_940 = True, ds_lut = None, verbose = True):
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
    dsmet = xr.open_dataset(fnmet)

    
    #### Make the langleys
    si = atmspec.CombinedGlobalDiffuseDirect(ds.copy())
    sir = si.direct_normal_irradiation
    sir.settings_langley_airmass_limits = langley_airmass_limits
    out['sir'] = sir
    # apply cloudmask
    # if not isinstance(lt, type(None)):
    #     sir.dataset = sir.dataset.where(cloudmask == 0)

    if test:
        return out
    if not clean:
        # print('no clean')
        out['langley_am'] = sir.langley_am
        out['langley_pm'] = sir.langley_pm
        pass
    elif clean == 'normal':
        lang = sir.langley_am
        
        # lang.plot(wavelength=None, show_pre_clean=False)
        # clean the langley
        lc = lang.clean(threshold=2)
        langc_am = lc['langley']
        
        lang = sir.langley_pm
        lc = lang.clean(threshold=2)
        langc_pm = lc['langley']
        
        out['langley_am'] = langc_am
        out['langley_pm'] = langc_pm
    elif clean == 'cloudscreening':
        assert(lt is not None), 'a previous calibration has to be past to lt for this to work, this could be done by running this first with a normal clean'
        # calibrate with preliminary calibration (the last ones)
        # the following is solely to create a cloudmask tha can be applied before performing the langley calibration
        gdd = atmspec.CombinedGlobalDiffuseDirect(ds.copy())
        gdd.path2solar_spectrum = '/Users/htelg/fundgrube/reference_data/solar/spectrum/solar_spectral_irradiance_e490_00a_amo.nc'
        gdd.dataset['channel_wavelength'] = gdd.dataset.channel_wavelength.astype(float) #error in processing upstream, fix it in future versions!
        gddc = gdd.apply_calibration_langley(lt)
    
        dnic = gddc.direct_normal_irradiation
        out['dnic'] = dnic
        dnic.dataset = dnic.dataset.where(dnic.dataset.channel < 1000, drop = True)
        dnic.met_data = dsmet
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
        sir.dataset = sir.dataset.where(cloudmask == 0)
        out['langley_am'] = sir.langley_am
        out['langley_pm'] = sir.langley_pm
    if cal_940:
        langts = atmlc.open_langleys([out['langley_am'].to_xarray_dataset()])
        pwvcal = determine_940_calcoeffs(sir.dataset, langts, dsmet, dsmet.ozone, ds_lut, ampm = 'am', amf_range = langley_airmass_limits, verbose = verbose)
        out['pwvcal_am'] = pwvcal
        
        langts = atmlc.open_langleys([out['langley_pm'].to_xarray_dataset()])
        pwvcal = determine_940_calcoeffs(sir.dataset, langts, dsmet, dsmet.ozone, ds_lut, ampm = 'pm', amf_range = langley_airmass_limits, verbose = verbose)
        out['pwvcal_pm'] = pwvcal
    return out

# now get the pwv calibretion
def open_pwvcals(p2fld, pattern = '*.nc'):
    """This function is designed to open all pwv cal files within a folder.
    Parameters
    ----------
    p2fld : Union[str, pl.Path, list]
        This can be either a path to a folder containing the pwv cal files or a list of paths 
        to the pwv cal files or a list of already opened pwv cal datasets.
    """
    pattern = 'pwv*.nc'
    if  isinstance(p2fld, (str, pl.Path)):
        p2fld = pl.Path(p2fld)
        p2flist = list(p2fld.glob(pattern))
    elif isinstance(p2fld, list):
        if all([isinstance(p, xr.Dataset) for p in p2fld]):
            p2flist = p2fld
        else:
            p2flist = [pl.Path(p) for p in p2fld]

    p2flist.sort()
    dslist = []
    for p2f in p2flist:
        if isinstance(p2f, xr.Dataset):
            ds = p2f
        else:
            ds = xr.open_dataset(p2f)
        if 'date' in ds.attrs:
            dt = pd.to_datetime(ds.attrs['date'])
        else:
            dt = pd.to_datetime(p2f.name.split('_')[-1].split('.')[0])
        if 'ampm' in ds.attrs:
            ampm = ds.attrs['ampm']
        else:
            assert(False)
        if ampm == 'pm':
            dt += pd.to_timedelta(6, 'h')
        ds = ds.expand_dims(datetime = [dt])
        # ds = ds.expand_dims(ampm = [ampm])
        ds['ampm'] =  ({'datetime':[dt]}, [ampm])
        dslist.append(ds)
    ds = xr.concat(dslist, 'datetime', join='outer')
    ds = ds.sortby('datetime')
    return PwvCal_Timeseries(ds)


class PwvCal_Timeseries(object):
    def __init__(self, dataset):
        self.dataset = dataset
        # self.predict_until = self.dataset.datetime.values[-1] + pd.to_timedelta(60, 'days')
        # # self.daterange2predict = pd.date_range(start = self.dataset.datetime.values[0], end = end, freq='D')
        # self._v0rol = None
        # self._v0gam = None
        # self._v0 = None
        # self._ranked = None    

    # def rank_by(self, wl = 500, fit_results = 'intercept_stderr'):
    #     """Ranks the current dataset by the given values"""
    #     ds = self.dataset
    #     dssel = ds.langley_fitres.sel(fit_results = fit_results, wavelength = wl, drop = True)
    #     ds['ranked'] = dssel.rank('datetime')
    #     ds['ranked'].attrs['description'] = f'Ranked by langley_fitres {fit_results} at {wl}nm'
    #     return
    @property
    def V0_simple(self):
        """This simply returns the V0 based on all langley result in this object. Therefore, kick out what you don't want!"""
        dsout = xr.Dataset()
        v0 = self.dataset.fitres.sel(statistic = 'estimate', parameter = 'V0', drop = True)
        dsout['V0'] = v0.mean('datetime')
        dsout['V0_std'] = v0.std('datetime', ddof = 1) 
        dsout.V0_std.attrs['description'] = 'nbiased standard deviation, ddof = 1'
        ns = []
        osubs = []
        alpha = 0.05
        # n = self.dataset.fitres.sel(statistic = 'extimate').dropna('datetime').shape[0]
        # ns.append(n)
        n = self.dataset.fitres.sel(statistic = 'estimate').dropna('datetime').shape[0]
        chi2val = scipy.stats.chi2.ppf(alpha, df=n-1)
        osub = np.sqrt((n-1)/chi2val) # this is the one-sided upper bound factor with a 95% confidence
            
        dsout['number_of_cals'] = n
        dsout.number_of_cals.attrs['description'] = 'Number of calibrations used.'
        dsout['one_sided_upper_bound_factor_95conf'] = osub
        dsout.one_sided_upper_bound_factor_95conf.attrs['description'] = r'one-sided upper bound factor with a 95% confidence, mostly relevant for small number of langleys.'

        additional_uncertainty = 0.005 # ARM introduced this additional uncertainty to cover remaining uncertainties. In Michalsky 2001 this is reflected in the remaining uncertainty even in the MLO calibrations.
        dsout['OD_uncertainty'] = (dsout['one_sided_upper_bound_factor_95conf'] * dsout['V0_std'] / dsout['V0']) + additional_uncertainty
        dsout.OD_uncertainty.attrs['description'] = r'(V0_std / V0 * osub) + 0.005. osub: one-sided upper bound factor with a 95% confidence, mostly relevant for small number of langleys. This still needs to be multiplide by 1/AMF to get the uncertainty in (A)OD. Michalsky 2001.'
        dsout['V0_stderr'] = self.dataset.fitres.sel(statistic = 'uncertainty', parameter = 'V0', drop = True).mean('datetime')
        return dsout
    def remove_V0_with_large_uncertainties(self, quantile = 0.9):
        """Note, apply this first before using remove_V0_outliers
        Removes data where the uncertainty of V0 is larger than the given quantile. """
        
        ds = self.dataset.copy()    
        stderr = ds.fitres.sel(statistic = 'uncertainty', parameter = 'V0')
        ds = ds.where(stderr < stderr.quantile(quantile, dim = 'datetime'))
        ds['fitres'] = ds.fitres.dropna('datetime')
        out = PwvCal_Timeseries(ds)
        return out

    def remove_V0_outliers(self, lower_quantile = 0.1, upper_quantile = 0.9):
        """Note, apply this after using remove_V0_with_large_uncertainties to get rid of the most uncertain ones first.
        Removes data where the estimate of V0 is outside of the given quantile range"""
        ds = self.dataset.copy()
        ds = ds.sortby(ds.fitres.sel(statistic = 'uncertainty', parameter = 'V0'))    
        upper = ds.fitres.sel(statistic = 'estimate', parameter = 'V0').quantile(upper_quantile, dim = 'datetime')
        lower = ds.fitres.sel(statistic = 'estimate', parameter = 'V0').quantile(lower_quantile, dim = 'datetime')
        dat= ds.fitres.sel(statistic = 'estimate', parameter = 'V0')
        ds = ds.where(np.logical_and(dat<upper, dat>lower))
        ds['fitres'] = ds.fitres.dropna('datetime')
        out = PwvCal_Timeseries(ds)
        return out

class PwvCalibration940(object):
    def __init__(self):
        self._fitres = None
        self.pwvcal_pre_clean = None
        return
    
    def perform_fit(self):
        I_pwv_r = self.normalized_direct_normal
        x_fit = I_pwv_r.airmass.values
        I_fit = I_pwv_r.values # used to be I_fit_data ... find a better term
        lut_da = self.lookuptable.optical_depth

        def transmission(pwv):
            tau = lut_da.interp(airmass=x_fit, pwv=pwv).values
            return np.exp(-tau)
        
        def best_I0_for_pwv(pwv):
            t = transmission(pwv)
            return np.sum(I_fit * t) / np.sum(t**2)
        
        def cost(pwv):
            t = transmission(pwv)
            I0 = np.sum(I_fit * t) / np.sum(t**2)
            return np.sum((I_fit - I0 * t) ** 2)
        
        res = sp.optimize.minimize_scalar(
            cost,
            bounds=(lut_da.pwv.min().item(), lut_da.pwv.max().item()),
            method="bounded",
        )
        
        pwv_fit = res.x
        I0_fit = best_I0_for_pwv(pwv_fit)


        # get uncertainty
        tau_da = lut_da
        dtau_dpwv_da = lut_da.differentiate("pwv")
        
        def tau(pwv): #todo: this function is the same as above!!
            return tau_da.interp(airmass=x_fit, pwv=pwv).values
        
        def dtau_dpwv(pwv):
            return dtau_dpwv_da.interp(airmass=x_fit, pwv=pwv).values
        
        def transmission(pwv):
            return np.exp(-tau(pwv))
        
        def dtransmission_dpwv(pwv):
            t = transmission(pwv)
            return -t * dtau_dpwv(pwv)

        t = transmission(pwv_fit)
        dt = dtransmission_dpwv(pwv_fit)
        
        J = np.column_stack([
            t,
            I0_fit * dt,
        ])
        residual = I_fit - I0_fit * t
        rss = np.sum(residual**2)
        
        n = I_fit.size
        p = 2
        sigma2 = rss / (n - p)
        
        pcov = sigma2 * np.linalg.inv(J.T @ J)
        
        I0_std, pwv_std = np.sqrt(np.diag(pcov)) # these are 1 sigma!
        fitres = dict(V0 = I0_fit,pwv = res.x)
        umcertainty = dict(V0 = I0_std,pwv = pwv_std)
        df = pd.concat([pd.DataFrame(fitres, index = ['estimate']),
                        pd.DataFrame(umcertainty, index = ['uncertainty'])])
        df.columns.name = 'parameter' 
        df.index.name = 'statistic'
        self._fitres = df  
        # make the instance
        # pwvcal = PwvCalibration940()
        # pwvcal.lookuptable = ds_lut
        # pwvcal.normalized_direct_normal = I_pwv_r #todo: set attributes, units, ...
        # pwvcal.fitres = dict(V0 = I0_fit,
        #                     pwv = res.x)
        # pwvcal.fitres_uncertainty = dict(V0 = I0_std,
        #                     pwv = pwv_std)
        return 



    def to_xarray(self):
        ds = xr.Dataset()
        ds['fitres'] = self.fitres
        ds.attrs['ampm'] = self.ampm
        return ds

    @property
    def fitres(self):
        if self._fitres is None:
            self.perform_fit()
        return self._fitres

    @property
    def fit(self):
        tau_pwv = self.lookuptable.interp(pwv = self.fitres.loc['estimate', 'pwv']).optical_depth
        fit = self.fitres.loc['estimate', 'V0'] * np.exp(-tau_pwv)
        return fit
    
    @property
    def fit_match_data_airmasses(self):
        tau_pwv = self.lookuptable.interp(pwv = self.fitres.loc['estimate', 'pwv'], airmass = self.normalized_direct_normal.airmass).optical_depth
        fit = self.fitres.loc['estimate', 'V0'] * np.exp(-tau_pwv)
        return fit

    
    @property
    def residual(self):
        return self.normalized_direct_normal - self.fit_match_data_airmasses

    def plot(self, ax = None, label = None, textpos = (0.6, 0.8), show_preclean = True, data_kwargs = {}, fit_kwargs = {}):
        if ax is None:
            f,a = plt.subplots()
        else:
            a = ax
            f = ax.get_figure()

        # tau_pwv = self.lookuptable.interp(pwv = self.fitres['pwv']).optical_depth
        # fit = self.fitres['V0'] * np.exp(-tau_pwv)
        # f,a = plt.subplots()
        if label is not None:
            labeld = label + ' data'
            labelf = label + ' fit'
        else:
            labeld = 'data'
            labelf = 'fit'
        if 'ls' not in data_kwargs and 'linestyle' not in data_kwargs:
            data_kwargs['ls'] = ''
        if 'marker' not in data_kwargs:
            data_kwargs['marker'] = '.'
        self.normalized_direct_normal.plot(ax = a, **data_kwargs, label = labeld)
        g = a.get_lines()[-1]
        col = g.get_color()
        if show_preclean and self.pwvcal_pre_clean is not None:
            self.pwvcal_pre_clean.normalized_direct_normal.plot(ax = a, color = col, alpha = 0.5, lw = 4,label = labeld + ' pre-clean', zorder = 0)

        fit = self.fit
        fit.plot(ax = a, **fit_kwargs, label = labelf)
        # g = a.get_lines()[-1]
        # g.set_label(labelf)

        a.set_xlim(-1, 6)
        pad = 1.05
        ymin = self.normalized_direct_normal.min() * 1/pad
        ymax = self.normalized_direct_normal.max() * pad 
        a.set_ylim(ymin, ymax)
        xmin = self.normalized_direct_normal.airmass.min() * 1/pad
        xmax = self.normalized_direct_normal.airmass.max() * pad
        a.set_xlim(xmin, xmax)
        # ylabel = r"Direct normal normalized to $\tau_{R}$ and $\tau{AOD}$"
        ylabel = r"Normalized $I_\mathrm{dir}$"
        ylabel = r"$I_\mathrm{dir}$ (V) normalized to $\tau_R$ and $\tau_\mathrm{AOD}$"
        a.set_ylabel(ylabel)
        if label is not None:
            txt = (f'{label}:\n')
        else:
            txt = ''
        txt += (f'V0: {self.fitres.loc["estimate", "V0"]:0.1f} ± {self.fitres.loc["uncertainty", "V0"]:0.1f}\n'
               f'pwv: {self.fitres.loc["estimate", "pwv"]:0.3f} ± {self.fitres.loc["uncertainty", "pwv"]:0.3f}')
        a.text(textpos[0], textpos[1], txt, transform = a.transAxes)
        a.legend()
        return f,a
    
    def clean(self, threshold = 3, verbose = False):
        scale = threshold
        pwvcal = copy.deepcopy(self)
        pwvcal.pwvcal_pre_clean = self
        # converged = False
        status = 'not convergent'
        for i in range(20):
            lfr = pwvcal.residual
                
            if lfr.dropna('airmass').shape[0] == 0: #fit failed from the beginning
                status = 'fit failed'
                break
            skewness = (lfr.mean()-lfr.median())/lfr.std()
            # print(f'skewness:\t{skewness:0.4f}')
            # print(f'lfr.mean():\t{lfr.mean():0.4f} {lfr.std():0.4f}')
            # print(f'lfr.median():\t{lfr.median():0.4f} {lfr.mad():0.4f}')
            # print(f'corrdet:\t{spi.am.langley_residual_correlation_prop["determinant"]:0.5f}')
            
            # print(f'mad vs std: {lfr.mad():0.3f} vs {lfr.std():0.3f} ... {lfr.mad()/lfr.std():0.5f}')
            # skescale = scale + (abs(skewness) * 5)
            skescale = scale * (1 + (scale * abs(skewness)))

            print(f'skewness: {skewness:0.4f}\t skewscale:{skescale:0.4f}')
            cutoff = (lfr.median() - (lfr.std() *skescale), lfr.median() + (lfr.std() *skescale))
            cut = 'both'
            if cut == 'below':
                where = lfr > cutoff[0]
            elif cut == 'above': # this is not used yet
                where = lfr < cutoff[1]
            elif cut == 'both':
                where = np.logical_and(lfr > cutoff[0], lfr < cutoff[1])

            if (~where).sum() == 0:
                # converged = True
                status = 'converged'
                break

            try:
                pwvcal.normalized_direct_normal = pwvcal.normalized_direct_normal[where]
            except:
                # print('possible reason for failure: there are nans in the langley data?')
                raise
            pwvcal.perform_fit()
                
            # spi.am._langley_fitres = None
            # spi.am._langley_residual_correlation_prop = None
            # spi.am._langley_fit_residual = None
            # break

        out = {'pwvcal': pwvcal}
        out['iterations'] = i
        out['status'] = status
        return out

def determine_940_calcoeffs(ds, langts, dsmet, ozone, ds_lut, ampm, amf_range, verbose = True):
    # todo: test all units
    out = {}
    gdd = atmspec.CombinedGlobalDiffuseDirect(ds.copy(), verbose = verbose)
    # apply langleys
    gdd = gdd.apply_calibration_langley(langts)

    ### Moving to direct
    dni = gdd.direct_normal_irradiation
    dni.dataset.attrs['serial_no'] = dni.dataset.sn_mfrsr # todo: this test should not be happening here, but in an external processing step (in surfradpy?)
    out['dni_inst'] =  dni

    # met data
    dni.met_data = dsmet

    # ozone
    # dni.ozone_data = 300
    ozone = ozone.where(ozone != -99.999)
    ozone = ozone.interpolate_na('time', method = 'nearest', fill_value = 'extrapolate')
    dni.ozone_data = ozone

    dni.skip_1625_channel = True

    dni.aod
    

    #extrapolate to get aod at 940 
    #todo, this needs to be added to the retrieval
    
    l1 =  500
    l2 = 870
    t1 = dni.aod.sel(channel = l1)
    t2 = dni.aod.sel(channel = l2)
    t1 = t1.where(t1>0, other = 0.001)
    t2 = t2.where(t2>0, other = 0.001)
    ang = - np.log(t1/t2)/np.log(l1/l2)
    l3 = 940
    t3 = t1 * (l3/l1)**-ang
    
    dni.aod.loc[dict(channel = 940)] = t3
    
    wl = 940
    dst = dni.dataset.copy()
    dst['direct_normal_raw'] = ds.direct_normal
    dst = dst.sel(channel = wl)
    dst = dst.swap_dims({'datetime':'airmass'})
    dst = dst.sortby('airmass')
    
    
    
    assert(dst.solar_azimuth_angle.units == 'degrees')
    
    if ampm == 'pm':
        dst = dst.where(dst.solar_azimuth_angle > 180, drop = True)
    elif ampm == 'am':
        dst = dst.where(dst.solar_azimuth_angle < 180, drop = True)
    
    dst = dst.where(np.logical_and(amf_range[0] < dst.airmass, dst.airmass < amf_range[1]), drop = True)
    
    dst = dst.where(np.isfinite(dst.airmass), drop = True)
    
    I_pwv_r = dst.direct_normal_raw/(np.exp(-dst.od_rayleigh) * np.exp(-dst.aod))
    
    assert(not (~np.isfinite(dst.od_rayleigh)).sum() > 0)
    assert(not (~np.isfinite(dst.aod)).sum() > 0)
    assert(not (~np.isfinite(dst.direct_normal)).sum() > 0)
    assert(not (~np.isfinite(I_pwv_r)).sum() > 0)

    
    # make the instance
    pwvcal = PwvCalibration940()
    pwvcal.lookuptable = ds_lut
    pwvcal.normalized_direct_normal = I_pwv_r
    pwvcal.ampm = ampm
    return pwvcal






    x_fit = I_pwv_r.airmass.values
    I_fit = I_pwv_r.values # used to be I_fit_data ... find a better term
    lut_da = ds_lut.optical_depth


    def transmission(pwv):
        tau = lut_da.interp(airmass=x_fit, pwv=pwv).values
        return np.exp(-tau)
    
    def best_I0_for_pwv(pwv):
        t = transmission(pwv)
        return np.sum(I_fit * t) / np.sum(t**2)
    
    def cost(pwv):
        t = transmission(pwv)
        I0 = np.sum(I_fit * t) / np.sum(t**2)
        return np.sum((I_fit - I0 * t) ** 2)
    
    res = sp.optimize.minimize_scalar(
        cost,
        bounds=(lut_da.pwv.min().item(), lut_da.pwv.max().item()),
        method="bounded",
    )
    
    pwv_fit = res.x
    I0_fit = best_I0_for_pwv(pwv_fit)


    # get uncertainty
    tau_da = lut_da
    dtau_dpwv_da = lut_da.differentiate("pwv")
    
    def tau(pwv): #todo: this function is the same as above!!
        return tau_da.interp(airmass=x_fit, pwv=pwv).values
    
    def dtau_dpwv(pwv):
        return dtau_dpwv_da.interp(airmass=x_fit, pwv=pwv).values
    
    def transmission(pwv):
        return np.exp(-tau(pwv))
    
    def dtransmission_dpwv(pwv):
        t = transmission(pwv)
        return -t * dtau_dpwv(pwv)

    t = transmission(pwv_fit)
    dt = dtransmission_dpwv(pwv_fit)
    
    J = np.column_stack([
        t,
        I0_fit * dt,
    ])
    residual = I_fit - I0_fit * t
    rss = np.sum(residual**2)
    
    n = I_fit.size
    p = 2
    sigma2 = rss / (n - p)
    
    pcov = sigma2 * np.linalg.inv(J.T @ J)
    
    I0_std, pwv_std = np.sqrt(np.diag(pcov)) # these are 1 sigma!
    
    # make the instance
    pwvcal = PwvCalibration940()
    pwvcal.lookuptable = ds_lut
    pwvcal.normalized_direct_normal = I_pwv_r #todo: set attributes, units, ...
    pwvcal.fitres = dict(V0 = I0_fit,
                         pwv = res.x)
    pwvcal.fitres_uncertainty = dict(V0 = I0_std,
                         pwv = pwv_std)
    return pwvcal



    
