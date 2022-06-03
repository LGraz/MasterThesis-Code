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
mkdir -p data/{computation_results,cv_itpl_res,pixels_pkl,scl}

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
-------------------------------------------------------------
                Interpolation
-------------------------------------------------------------
"

my_python () {
  echo -ne "
  -----------------------------------------
  --   execute:  $1 
  "
  python $1
  echo -ne "-- done
  "
}

## plots
my_python "./interpol/methods/fourier_plots.py"
my_python "./interpol/methods/kriging_plots.py"
my_python "./interpol/methods/problem_illustration.py"
# my_python "./interpol/methods/loess_plots.py"
my_python "./interpol/methods/cv/plot_res_cv.py"
my_python "./interpol/scl_plots.py"

## parameter estimation
my_python "./interpol/methods/cv/cv_itpl_res.py"


echo -ne "
-------------------------------------------------------------
                
-------------------------------------------------------------
"


echo -ne "
-------------------------------------------------------------
                
-------------------------------------------------------------
"
# if on stats cluster copy computation results s.t. we can reach them via scp ...
if [ $thesis_dir = /userdata/lgraz ]; then 
rm ~/computation_results
cp $thesis_dir/code/data/computation_results ~/computation_results
fi

echo -ne "
-------------------------------------------------------------
                Latex build
-------------------------------------------------------------
"
mkdir $latex_dir
cd $latex_dir
echo "TODO pdflatex ..."

echo -ne "
-------------------------------------------------------------
                Hurray you finished
-------------------------------------------------------------
"
