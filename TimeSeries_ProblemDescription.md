## Goal: 
estimate yield for each pixel of the field (at the resolution of satellite-image 10x10m)  

## What is given
Every pixel is one data-point. In total we have approx 80000 such pixels (to remove autocorrelation one might consider only alternating pixels) 
For *each* pixel we have:  
- **the yield:**  
    a number
- **weather time series data:**   
    for *each day* multiple variables like: humidity, sunshine-hours, rainfall). Note that we have basically the same weather for nearby pixels
- **light reflectance time series:**  
    for each time point: reflectance at 12 frequencies (from UV to infrared). Note that time points are not equidistant since we filter out clouds. Different time points filtered for different pixels.

## Challenge
So basically we have *for each point* one target variable (yield) and multiple time series.   
How can we estimate the yield?  
How do we deal with the non-equidistance of the light-reflectances-time-series?

## Ideas:
1. Use splines to interpolate the time series of light reflectance, such that we have an estimate for every day
2. Substitute each time series with several descriptive parameters (like: peak, integral_1_half, integral_2_half, ...) and now form a table like:  
Pixel_1 | yield | parameters_TimeSeries_1 | parameters_TimeSeries_2 | ...  
Pixel_2 | yield | parameters_TimeSeries_1 | parameters_TimeSeries_2 | ...  
...  
now apply usual regression methods: (lasso / additive models / MARS / Trees or RandomForest / OLS / GLM)
3. from a paper:  
    Estimate a time series of the Biomass.   
    Intuition: Each day were there is good weather (and the satellite images get greener) the plant grows.   
    In the end multiply total biomass (for each pixel) by species-dependent factor to estimate yield.

