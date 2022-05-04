#!/usr/bin/bash
Mc=/home/lukas/Documents/ETH/MASTER_THESIS/code
cp -r $Mc $Mc/

# remove pkl files for size
find . -name $Mc/code/data/yieldmapping_data "*.pkl" -type f -delete
rm -rf $Mc/code/.env

#copy files to remote
scp -r $Mc/code lgraz@sftpmath.math.ethz.ch:/home/thesis/

rm -rf $Mc/code
