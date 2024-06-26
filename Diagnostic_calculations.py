from global_land_mask import globe
import xarray as xr
import numpy as np

def WCTH_calc(da,latranges):
    '''
    This function calculates the WCTH from an xarray dataarray
    da should have dimensions tau, cth, lat, and lon.
    any other dimensions such as time will not be reduced.
    latranges should be a dictionary where the keys are the names of
    each latitude range corrosponding to each cth threshold
    and the elements are the latitude bounds. 
    '''
    lat_keys = list(latranges.keys())

    WCTH_lat_list = []
    for lr in (lat_keys):  # This is to accomodate cth thresholds that vary with latitude
        CF_cth = da.sel(lat=slice(latranges[lr][0],latranges[lr][1])).sel(cth=slice(latranges[lr][2],latranges[lr][3])).sum('tau')
        ## Now calculate WCTH
        WCTH_num = CF_cth*CF_cth.cth
        WCTH_den = CF_cth.sum('cth')
        WCTH_local = WCTH_num.sum('cth')/WCTH_den
        ## Append them to the lists
        WCTH_lat_list.append(WCTH_local)

    WCTH_latlon = xr.concat(WCTH_lat_list,'lat').sortby('lat')
    return(WCTH_latlon)
################################################################################

def WTAU_calc(da,latranges):
    '''
    This function calculates the WTAU from an xarray dataarray
    da should have dimensions tau, cth, lat, and lon.
    any other dimensions such as time will not be reduced.
    latranges should be a dictionary where the keys are the names of
    each latitude range corrosponding to each cth threshold
    and the elements are the latitude bounds. 
    '''
    lat_keys = list(latranges.keys())

    WTAU_lat_list = []
    for lr in (lat_keys):  # This is to accomodate cth thresholds that vary with latitude
        CF_tau = da.sel(lat=slice(latranges[lr][0],latranges[lr][1])).sel(cth=slice(latranges[lr][2],latranges[lr][3])).sum('cth')
        ## Now calculate WTAU
        WTAU_num = CF_tau*CF_tau.tau
        WTAU_den = CF_tau.sum('tau')
        WTAU_local = WTAU_num.sum('tau')/WTAU_den
        ## Append them to the lists
        WTAU_lat_list.append(WTAU_local)

    WTAU_latlon = xr.concat(WTAU_lat_list,'lat').sortby('lat')
    return(WTAU_latlon)
################################################################################

def CF_calc(da,latranges):
    '''
    This function calculates the CF from an xarray dataarray
    da should have dimensions tau, cth, lat, and lon.
    any other dimensions such as time will not be reduced.
    latranges should be a dictionary where the keys are the names of
    each latitude range corrosponding to each cth threshold
    and the elements are the latitude bounds. 
    '''
    
    lat_keys = list(latranges.keys())

    CF_lat_list = []
    for lr in (lat_keys): # This is to accomodate cth thresholds that vary with latitude
        CF_cth = da.sel(lat=slice(latranges[lr][0],latranges[lr][1])).sel(cth=slice(latranges[lr][2],latranges[lr][3])).sum('cth').sum('tau')
        ## Append them to the lists
        CF_lat_list.append(CF_cth)

    CF_latlon = xr.concat(CF_lat_list,'lat').sortby('lat')
    return(CF_latlon)
################################################################################


def CF_wgt_regional_mean(da,da_CF,ocean_only=False):
    ## this function works averages over any latitude and longitude range for an xarray dataarray
    ## da_CF must not have a time dimension
    lat = da.lat
    lon = da.lon
    longrid,latgrid = np.meshgrid(lon,lat)
    if ocean_only == True:
        oceanmask = xr.DataArray(globe.is_ocean(latgrid,longrid),dims=['lat','lon'])
        oceanmask = oceanmask.where(oceanmask==True)
        gw_base = np.transpose(np.cos(np.asarray(lat)*np.pi/180))
        gw = np.transpose(np.tile(gw_base,(len(lon),1))) * oceanmask
    else:
        gw_base = np.transpose(np.cos(np.asarray(lat)*np.pi/180))
        gw = np.transpose(np.tile(gw_base,(len(lon),1)))
    gw = xr.DataArray(gw,dims=['lat','lon'])*da_CF
    gw = gw/gw.sum()
    da_gw = da * gw
    out = da_gw.sum('lat').sum('lon')
    return(out)
################################################################################

def regional_mean(da,ocean_only=False):
    ## this function works averages over any latitude and longitude range for an xarray dataarray
    lat = da.lat
    lon = da.lon
    longrid,latgrid = np.meshgrid(lon,lat)
    if ocean_only == True:
        oceanmask = xr.DataArray(globe.is_ocean(latgrid,longrid),dims=['lat','lon'])
        oceanmask = oceanmask.where(oceanmask==True)
        gw_base = np.transpose(np.cos(np.asarray(lat)*np.pi/180))
        gw = np.transpose(np.tile(gw_base,(len(lon),1))) * oceanmask
    else:
        gw_base = np.transpose(np.cos(np.asarray(lat)*np.pi/180))
        gw = np.transpose(np.tile(gw_base,(len(lon),1)))
    gw = xr.DataArray(gw,dims=['lat','lon'])
    gw = gw/gw.sum()
    da_gw = da * gw
    out = da_gw.sum('lat').sum('lon')
    return(out)
################################################################################

def lon_m180_to_p180(ds):
    lon = ds.lon
    if lon.max()>180:
        ds = ds.assign_coords({'lon':(lon+180)%360-180}).sortby('lon').sortby('lat')
    else:
        ds = ds.sortby('lon').sortby('lat')
    return(ds)
################################################################################

def CF_descent(ds_misr,ds_wap,lats,cth_latranges,taurange,wap_threshold=10):
    '''
    Works one model and lat chunk at a time
    lr must be either two latitudes to define the range, or one where the range crosses the equator.
    I.E. [30] for -30 to +30, or [40,60] for 40 to 60 degrees.
    ds_wap should already be converted into hPa/day. Alternatively, could just convert the wap threshold.
    '''
    taumin = taurange[0]
    taumax = taurange[1]
    
    Metric_vals = xr.Dataset()
    ds = ds_misr.interp({'lat':ds_wap.lat,'lon':ds_wap.lon}) #ensure that wap and clmisr are on the same grid
    # cut the clmisr data to fit the correct latitude range    
    if len(lats) == 1:
        latmin = -1*lats[0]
        latmax = lats[0]
        da_full = ds.sel(lat=slice(latmin,latmax)).clmisr.where(ds_wap.wap>wap_threshold).load() ## full means no reduced dimensions
    elif len(lats) == 2:
        latmin = lats[0]
        latmax = lats[1]
        da_NH = ds.sel(lat=slice(latmin,latmax)).clmisr.where(ds_wap.wap>wap_threshold).load()
        da_SH = ds.sel(lat=slice(-1*latmax,-1*latmin)).clmisr.where(ds_wap.wap>wap_threshold).load()
        da_full = xr.concat([da_SH,da_NH],'lat') ## full means no reduced dimensions
    da_full = da_full.where(da_full>0,0) # remove any possible fill values
    
    da = da_full.sel(tau=slice(taumin,taumax))
    ranges = cth_latranges # renaming this for simplicity
    # Calculate the metric
    CF = CF_calc(da,ranges)
    ## take the regional mean of each metric
    CF_rm = regional_mean(CF,ocean_only=True)

    CF_mean = CF_rm
    Metric_vals['CF'] = CF_mean
    return(Metric_vals)

def WCTH_regional(ds_misr,lats,cth_latranges,taurange):
    '''
    Works one model and lat chunk at a time
    lr must be either two latitudes to define the range, or one where the range crosses the equator.
    I.E. [30] for -30 to +30, or [40,60] for 40 to 60 degrees.
    ds_wap should already be converted into hPa/day. Alternatively, could just convert the wap threshold.
    '''
    taumin = taurange[0]
    taumax = taurange[1]
    
    Metric_vals = xr.Dataset()
    ds = ds_misr
    # cut the clmisr data to fit the correct latitude range    
    if len(lats) == 1:
        latmin = -1*lats[0]
        latmax = lats[0]
        da_full = ds.sel(lat=slice(latmin,latmax)).clmisr.load() ## full means no reduced dimensions
    elif len(lats) == 2:
        latmin = lats[0]
        latmax = lats[1]
        da_NH = ds.sel(lat=slice(latmin,latmax)).clmisr.load()
        da_SH = ds.sel(lat=slice(-1*latmax,-1*latmin)).clmisr.load()
        da_full = xr.concat([da_SH,da_NH],'lat') ## full means no reduced dimensions
    da_full = da_full.where(da_full>0,0) # remove any possible fill values
    
    da = da_full.sel(tau=slice(taumin,taumax))
    ranges = cth_latranges # renaming this for simplicity
    # Calculate the metric
    WCTH = WCTH_calc(da,ranges)
    CF = CF_calc(da,ranges)
    # Check if 'time' is in CF and average over time if necessary
    CF_dims = list(CF.dims)
    if 'time' in (CF_dims):
        CF = CF.mean('time')
    ## take the regional mean of each metric
    WCTH_rm = CF_wgt_regional_mean(WCTH,CF,ocean_only=True)

    WCTH_mean = WCTH_rm
    Metric_vals['WCTH'] = WCTH_mean
    return(Metric_vals)


def WTAU_regional(ds_misr,lats,cth_latranges,taurange):
    '''
    Works one model and lat chunk at a time
    lats must be either two latitudes to define the range, or one where the range crosses the equator.
    I.E. [30] for -30 to +30, or [40,60] for 40 to 60 degrees.
    ds_wap should already be converted into hPa/day. Alternatively, could just convert the wap threshold.
    '''
    taumin = taurange[0]
    taumax = taurange[1]
    
    Metric_vals = xr.Dataset()
    ds = ds_misr
    # cut the clmisr data to fit the correct latitude range    
    if len(lats) == 1:
        latmin = -1*lats[0]
        latmax = lats[0]
        da_full = ds.sel(lat=slice(latmin,latmax)).clmisr.load() ## full means no reduced dimensions
    elif len(lats) == 2:
        latmin = lats[0]
        latmax = lats[1]
        da_NH = ds.sel(lat=slice(latmin,latmax)).clmisr.load()
        da_SH = ds.sel(lat=slice(-1*latmax,-1*latmin)).clmisr.load()
        da_full = xr.concat([da_SH,da_NH],'lat') ## full means no reduced dimensions
    da_full = da_full.where(da_full>0,0) # remove any possible fill values
    
    da = da_full.sel(tau=slice(taumin,taumax))
    ranges = cth_latranges # renaming this for simplicity
    # Calculate the metric
    WTAU = WTAU_calc(da,ranges)
    CF = CF_calc(da,ranges)
    # Check if 'time' is in CF and average over time if necessary
    CF_dims = list(CF.dims)
    if 'time' in (CF_dims):
        CF = CF.mean('time')
    
    ## take the regional mean of each metric
    WTAU_rm = CF_wgt_regional_mean(WTAU,CF,ocean_only=True)

    WTAU_mean = WTAU_rm
    Metric_vals['WTAU'] = WTAU_mean
    return(Metric_vals)


def CF_regional(ds_misr,lats,cth_latranges,taurange):
    '''
    Works one model and lat chunk at a time
    lr must be either two latitudes to define the range, or one where the range crosses the equator.
    I.E. [30] for -30 to +30, or [40,60] for 40 to 60 degrees.
    ds_wap should already be converted into hPa/day. Alternatively, could just convert the wap threshold.
    '''
    taumin = taurange[0]
    taumax = taurange[1]
    
    Metric_vals = xr.Dataset()
    ds = ds_misr
    # cut the clmisr data to fit the correct latitude range    
    if len(lats) == 1:
        latmin = -1*lats[0]
        latmax = lats[0]
        da_full = ds.sel(lat=slice(latmin,latmax)).clmisr.load() ## full means no reduced dimensions
    elif len(lats) == 2:
        latmin = lats[0]
        latmax = lats[1]
        da_NH = ds.sel(lat=slice(latmin,latmax)).clmisr.load()
        da_SH = ds.sel(lat=slice(-1*latmax,-1*latmin)).clmisr.load()
        da_full = xr.concat([da_SH,da_NH],'lat') ## full means no reduced dimensions
    da_full = da_full.where(da_full>0,0) # remove any possible fill values
    
    da = da_full.sel(tau=slice(taumin,taumax))
    ranges = cth_latranges # renaming this for simplicity
    # Calculate the metric
    CF = CF_calc(da,ranges)
    ## take the regional mean of each metric
    CF_rm = regional_mean(CF,ocean_only=True)
    
    CF_mean = CF_rm
    Metric_vals['CF'] = CF_mean
    return(Metric_vals)
