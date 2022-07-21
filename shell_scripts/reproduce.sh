#!/usr/bin/bash

start_time=`date +%s.%N` 
echo -ne "
==================================================================
                Set up: Directory & Python & Latex
==================================================================
"
## Working Directories
echo -ne "
Are you in the 'thesis_dir' directory? if so please press enter,
else: type directory: "
read thesis_dir
if [ -z "$thesis_dir" ]; then 
    thesis_dir=$PWD; 
else 
    echo $thesis_dir
    cd $thesis_dir || { echo 'Bad directory, specify the whole path, exiting script ...' ; exit 1; }
fi
echo "you are in $thesis_dir"
code_dir=${thesis_dir}/code
latex_dir=${thesis_dir}/latex

## Get 'latex' form github
if [ ! -d "$latex_dir" ] 
then
    echo "TODO: git clone ..." 
    exit 1
fi

## Get 'code' form github
if [ ! -d "$code_dir" ] 
then
    echo "TODO: git clone ..." 
    exit 1
fi

## Python setup
echo -ne "
set up pyton venv"
cd $code_dir
if [[ ! -d "./.env" ]] ; then
    python -m venv ./.env || { echo 'failed to generate virtual python ".env", exiting script ...' ; exit 1; }
fi
echo "activate venv"
source .env/bin/activate || { echo 'failed to activate virtual python ".env", exiting script ...' ; exit 1; }
echo "installing requirements"
pip install -r requirements.txt
pip install wcwidth


echo -ne "
==================================================================
                        Data
==================================================================
"
mkdir -p data/computation_results/{cv_itpl_res,pixels_pkl,scl,ndvi_tables,ml_models/R,pixels_itpl_corr_dict_array}

## check yieldmapping data
if [[ ! -d "./data/yieldmapping_data/cloudy_data/yearly_train_test_sets" ]] ; then
    echo -ne "put yieldmapping_data at the location './data/yieldmapping_data'
    it can be downloaded from: https://polybox.ethz.ch/index.php/s/dBvfgSpOYsi3MUP"
    sleep 3
    echo "press ENTER to continiue"
    read dummy
    if [[ ! -d "./data/yieldmapping_data/cloudy_data/yearly_train_test_sets" ]] ; then
        echo "failed to locate data"
        exit 1
    fi
fi

echo -ne "
==================================================================
                    M I S C
==================================================================
"
# get satelite-ts-plot
my_python "./plots_witzwil/s2_field_timeseries.py" #&


echo -ne "
==================================================================
                    Interpolation
==================================================================
"

my_python () {
  echo -ne "
  -----------------------------------------
  --   execute:  $1 
  "
  time python $1
  echo -ne "-- done
  "
}

## plots
my_python "./interpol/methods/fourier_plots.py" #&
my_python "./interpol/methods/kriging_plots.py" #&

## parameter estimation
my_python "./interpol/methods/cv/cv_itpl_res.py"

## illustrate choice of statistic we optimize with respect to
my_python "./interpol/methods/plot_ss_loess.py" #&

echo -ne "
==================================================================
                    NDVI - Correction
==================================================================
"
# get table where each row is a time point of a pixel and contains 
# all information including interpolation values
my_python "./ndvi_corr/get_ndvi_table.py"

# illustrate that other scl_classes might still be useful
my_python "./ndvi_corr/scl_plots.py" #&

# simple ndvi-ts-plot of selected pixel, interpolation and scl_color
my_python "./ndvi_corr/residuals.py" #&

# train & analyze NDVI-correction Models (10%)
Rscript "./ndvi_corr/train_analyze_ndvi_correction.R"

# get stepwise illustration of how correction works
my_python "./ndvi_corr/plot_corrected.py" #&

# get corrected ndvi data (10%)
my_python "./ndvi_corr/get_corrected_table.py"

# evaluate w.r.t yield-predictability how good correction & robustification works (10%)
Rscript "./ndvi_corr/eval_correction_method.R"

echo -ne "
==================================================================

==================================================================
"
# if on stats cluster copy computation results s.t. we can reach them via scp ...
if [ $thesis_dir = /userdata/lgraz ]; then 
mkdir -p ~/thesis/code/data/
rm -rf ~/thesis/code/data/computation_results
cp $thesis_dir/code/data/computation_results ~/thesis/code/data/
fi

echo -ne "
==================================================================
                         Latex build
==================================================================
"
mkdir $latex_dir
cd $latex_dir
echo "TODO pdflatex ..."

echo -ne "
==================================================================
                    Hurray you finished
==================================================================
"

# print execution time
end_time=`date +%s.%N`
runtime=$( echo "$end_time - $start_time" | bc -l )
echo "Execution of everything needed $runtime seconds"
