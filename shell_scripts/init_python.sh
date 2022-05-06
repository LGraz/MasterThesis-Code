#!/usr/bin/bash

echo -ne "
-------------------------------------------------------------
                Set up: Directory & Python & Latex
-------------------------------------------------------------
"
# Working Directories
echo -ne "
Are you in the 'code' directory? if so please press enter, else interrupt
"
read dummy

# Python setup
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
mkdir data
mkdir data/computation_results

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
python data/data_manipulation/yielmapping_to_pickle.py
