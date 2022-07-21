# Summary of Project
-- 
# Overview over this Repository
Even though this repo contains only `thesis/code/` it works with the whole `thesis/`-directory. E.g. `thesis/latex/` is initialized during the initialization and figures are stored in `thesis/latex/figures`. The working directory for python/R code is `thesis/code/` unless specified otherwise.

## Directories - description
An description of the most important directories in `thesis/...`:

`code/data/` -- here all the data is stored  
`code/data/yieldmapping_data/` -- the original untouched data (compressed with pickle for faster loading)  
`code/data/computation_results/` -- results of computations (to avoid unnecessary computations each time)  
`code/data/data_description/` -- description of yieldmapping data   
`code/interpol/` -- all python scripts regarding the **interpolation**-chapter  
`code/my_utils/` -- help functions to be used in other scripts (by `import my_utils.<fname.py>`)  
`code/ndvi_corr/` -- scripts demonstrating and computing the NDVI-correction discussed in the chapter **NDVI-correction**  
`code/plots_witzwil/` -- plots related to satellite image view of witzwil  
`code/shell_scripts/` -- `bash`-scripts for efficient reproducibility  

`latex/` -- the `latex`-directory for writing the thesis  
`latex/figures` -- all figures used during the thesis (and the beamer-presentation)  
`latex/sty` -- style-files to give the ETH-look  
`latex/tex` -- all the written text visible in the thesis  

`beamer/` -- a `latex` (beamer) presentation  

## Pixel Class
Since we heavily focus on individual pixels and their time series we treat each pixel as an object (`my_utils.pixel.Pixel`)  

Functionalities of such a pixel object are:
- Providing the (corrected) NDVI-time-series
- The application of various *interpolation-methods* (e.g. smoothing splines; c.f. `my_utils.itpl`) with various *interpolation-strategies* (e.g. identity, cross-validation; c.f. `my_utils.strategies`), while applying various weighting/filtering methods
- Plot ndvi/interpolation results
- Parallelization: computations on a list of pixels can be parallelized by `my_utils.pixel_multiprocess.pixel_multiprocess(<list-of-pixels>)`  



# How to Reproduce
**Requirements:** A linux machine with recent versions of `R` and `python`
Also the python librarys `pandas` and `seaborn` should be installed for the user (for `R` to use it)

## Directory setup
```
thesis/code/data/yieldmapping_data/<data here>
```
now make sure that you are in the thesis directory, i.e. `cd .../thesis`

## Initialize
run: `./code/shell_scripts/init_environment.sh`
this initializes the python environment and get the newest version from GitHub 

## Run python jobs
run: `./code/shell_scripts/reproduce.sh`
