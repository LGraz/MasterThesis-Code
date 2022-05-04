#!/usr/bin/bash

echo -ne "
-------------------------------------------------------------
                Set up: Directory & Python & Latex
-------------------------------------------------------------
"
# Working Directories
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

# Get 'latex' form github
if [ ! -d "$latex_dir" ] 
then
    echo "TODO: git clone ..." 
    exit 1
fi

# Get 'code' form github
if [ ! -d "$code_dir" ] 
then
    echo "TODO: git clone ..." 
    exit 1
fi

# Python setup
echo -ne "
set up pyton venv
checking if .env exists: (y/n)"
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

# check yieldmapping data
if [[ ! -d "./data/yieldmapping_data/cloudy_data/yearly_train_test_sets" ]] ; then
    echo "put yieldmapping_data at the location './data/yieldmapping_data'"
    sleep 3
    echo "press ENTER to continiue"
    read dummy
    if [[ ! -d "./data/yieldmapping_data/cloudy_data/yearly_train_test_sets" ]] ; then
        echo "failed to locate data"
        exit 1
    fi
fi

# get pickel for each csv-file in ./data/yielmapping_data (only update)
python data/data_manipulation/yielmapping_to_pickle.py
echo -ne "
-------------------------------------------------------------
                Interpolation
-------------------------------------------------------------
"
# plots
python "./interpol/methods/fourier_plots.py"
python "./interpol/methods/kriging_plots.py"
# python "./interpol/methods/sav_golay.py"




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