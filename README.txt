This repository performs the calculations of Aerenson & Marchand 2024. It requires input of a netcdf file of MISR or MISR simulator CTH-OD histograms, and a field of 500 hPa pressure velocity.

Simply change the path in main.py to the desired netcdf files. The current data processing/unit conversions are set up for MISR observations and ERA5 pressure velocity, so some of that will need to be changed for model output. The end of main.py produces xarray objects that can either be saved or plotted.

For comparison, the metrics calculated with MISR observations and ERA5 pressure velocity (including the NR correction described in Aerenson & Marchand 2024) are as follows:

Tropical High Cloud WTAU: 15.069

Midlatitude Low Cloud Fraction: 49.258 %

Midlatitude Low Cloud WCTH: 1278.4 m

Global High Cloud Fraction: 11.335 %

Tropical Low Cloud Fraction in Regions of Strong Descent: 18.515


For corrospondence or questions please contact myself at taerenso@uwyo.edu

Dependencies: Global_land_mask, xarray, numpy.
