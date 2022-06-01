#!/usr/bin/bash
M=/home/lukas/Documents/ETH/MASTER_THESIS

rm -rf $M/temp
mkdir -p $M/temp/
cp -r $M/code $M/temp/

# remove pkl files for size
find $M/temp/code/data/yieldmapping_data -name  "*.pkl" -type f -delete
rm -rf $M/temp/code/.env

#copy files to remote
scp -rC $M/temp/code lgraz@sftpmath.math.ethz.ch:/home/thesis/

rm -rf $M/temp
