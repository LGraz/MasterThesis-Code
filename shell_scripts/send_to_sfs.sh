#!/usr/bin/bash
M=/home/lukas/Documents/ETH/MASTER_THESIS

rm -rf $M/temp
mkdir -p $M/temp/code
cp -r $M/code/{data,shell_scripts} $M/temp/

# convert to pickle and remove csv files for size
python $M/temp/code/data/data_manipulation/yielmapping_to_pickle.py
find $M/temp/code/data/yieldmapping_data -name  "*.csv" -type f -delete

#copy files to remote
scp -rC $M/temp/code lgraz@sftpmath.math.ethz.ch:/userdata/lgraz/thesis/

rm -rf $M/temp/code
