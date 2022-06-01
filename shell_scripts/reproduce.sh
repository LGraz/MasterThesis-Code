#!/usr/bin/bash

echo -ne "
-------------------------------------------------------------
                Set up: Directory & Python & Latex
-------------------------------------------------------------
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


echo -ne "
-------------------------------------------------------------
                    Data
-------------------------------------------------------------
"
mkdir data
mkdir data/computation_results
mkdir data/computation_results/cv_itpl_res
mkdir data/computation_results/pixels_pkl
mkdir data/computation_results/scl

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

## get pickel for each csv-file in ./data/yielmapping_data (only update)
python data/data_manipulation/yielmapping_to_pickle.py
echo -ne "
-------------------------------------------------------------
                Interpolation
-------------------------------------------------------------
"
## plots
python "./interpol/methods/fourier_plots.py"
python "./interpol/methods/kriging_plots.py"
python "./interpol/methods/problem_illustration.py"
# python "./interpol/methods/loess_plots.py"
python "./interpol/methods/cv/plot_res_cv.py"
python "./interpol/scl_plots.py"

## parameter estimation
python "./interpol/methods/cv/cv_itpl_res.py"


echo -ne "
-------------------------------------------------------------
                
-------------------------------------------------------------
"


echo -ne "
-------------------------------------------------------------
                
-------------------------------------------------------------
"


echo -ne "
-------------------------------------------------------------
                Latex build
-------------------------------------------------------------
"
cd $latex_dir
echo "TODO pdflatex ..."

echo -ne "
-------------------------------------------------------------
                Hurray you finished
-------------------------------------------------------------
"
