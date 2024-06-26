from global_land_mask import globe
import xarray as xr
import numpy as np
import Diagnostic_calculations as DC

## Read in the MISR observations
path = '[PATH TO MISR DATA]' # CHANGE ME
ds = xr.open_mfdataset(path,use_cftime=True,engine='netcdf4').sel(time = slice('2001-02-01','2014-12-30')).load().rename({'clMISR':'clmisr'})
dimlist = list(ds.dims)
if 'alt16' in dimlist:
    ds = ds.rename({'alt16':'cth'}) # rename the cth dimension to match CMIP6 models

ds_MISR = DC.lon_m180_to_p180(ds)  # ensure that all data has the same longitude definition


## Read in the 500 hPa vertical velocity
path = '[PATH TO PRESSURE VELOCITY DATA]' # CHANGE ME
ds_wap = xr.open_mfdataset(path,use_cftime=True,engine='netcdf4').sel(time = slice('2001-02-01','2014-12-30')).sel(level=500) # select 500 hPa level
da_corrected = ds_wap.omega.load()*60*60*24/100 # unit conversions to hPa/day
ds_wap = xr.Dataset({'wap':da_corrected}).drop('level')
ds_wap = DC.lon_m180_to_p180(ds_wap)
ds_wap = ds_wap.mean('time')


# The following 3 dictionaries define the high, midlevel, and low cth thresholds that vary with latitude.
highranges = {
    'NHmid':(40,90,5000,1000000),
    'NHsub':(20,40,7000,1000000),
    'trop':(-20,20,9000,1000000),
    'SHsub':(-40,-20,7000,1000000),
    'SHmid':(-90,-40,5000,1000000)
}

midranges = {
    'NHmid':(40,90,3000,5000),
    'NHsub':(20,40,3000,7000),
    'trop':(-20,20,3000,9000),
    'SHsub':(-40,-20,3000,7000),
    'SHmid':(-90,-40,3000,5000)
}


lowranges = {
    'NHmid':(40,90,0,3000),
    'NHsub':(20,40,0,3000),
    'trop':(-20,20,0,3000),
    'SHsub':(-40,-20,0,3000),
    'SHmid':(-90,-40,0,3000)
}

# Define the latitude ranges
lats_dic = {
    'Global':[90],
    'Tropical':[30],
    'Midlatitude':[30,60],
    'High_latitude':[40,70]
}

taurange = (0.3,10000000) # For all metrics we use the same taurange, where we remove all retrievals with OD less than 0.3 (which are unreliable)


# Tropical High cloud optical depth
Trop_High_OD = DC.WTAU_regional(ds_MISR,lats_dic['Tropical'],highranges,taurange)

# Midlatitude Low cloud fraction
Midlat_Low_CF = DC.CF_regional(ds_MISR,lats_dic['Midlatitude'],lowranges,taurange)

# Midlatitude Low cloud top height
Midlat_Low_WCTH = DC.WCTH_regional(ds_MISR,lats_dic['Midlatitude'],lowranges,taurange)

# Global High cloud fraction
Global_High_CF = DC.CF_regional(ds_MISR,lats_dic['Global'],highranges,taurange)

# Tropical low cloud fraction
Tropical_low_CF = DC.CF_descent(ds_MISR,ds_wap,lats_dic['Tropical'],lowranges,taurange)


# The above 5 xarray objects can either be saved with the xr.Dataset.to_netcdf() function or plotted as one would like.