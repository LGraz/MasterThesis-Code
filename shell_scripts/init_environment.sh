#!/usr/bin/bash

echo -ne "
-------------------------------------------------------------
                Set up: Directory & Python & Latex
-------------------------------------------------------------
"
# Working Directories
echo -ne "
Are you in the 'thesis' directory? if so please press enter, else interrupt
"
read dummy

# Github
mkdir temp
mv code temp; # initially code contatins `data` and `shell_scripts`
git clone git@github.com:Greeenstone/MasterThesis-Code.git code
mv temp/code/data code/; rm -rf temp
git clone git@github.com:Greeenstone/MasterThesis-Documentation.git latex

# Python setup
cd code
echo -ne "
set up pyton venv"
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
mkdir -p data/computation_results/{pixels_pkl,scl,cv_itpl_res}

# check yieldmapping data
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

# get pickel for each csv-file in ./data/yielmapping_data (only update)
python my_utils/data_processing/yielmapping_to_pickle.py
