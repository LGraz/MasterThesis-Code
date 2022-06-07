#!/usr/bin/bash
M=/home/lukas/Documents/ETH/MASTER_THESIS ; cd $M

# update requirements.txt file
cd code; pipreqs --force .; cd $M

# push to github
cd code
git add . ; git commit -m "auto-commit by 'send_to_sfs.sh'" ; git push origin
cd $M
cd latex
git add . ; git commit -m "auto-commit by 'send_to_sfs.sh'" ; git push origin
cd $M

# copy first into temp-directory
rm -rf $M/temp
mkdir -p $M/temp/code
cp -r $M/code/{data,shell_scripts} $M/temp/code/

# convert to pickle and remove csv files for size
cd code; python $M/temp/code/data/data_manipulation/yielmapping_to_pickle.py; cd ..
find $M/temp/code/data/yieldmapping_data -name  "*.csv" -type f -delete
# remove unneccecary data
rm code/data/yieldmapping_data/*.pkl  # all years combined data

#copy files to remote
scp -rC $M/temp/code lgraz@sftpmath.math.ethz.ch:/home/thesis/

rm -rf $M/temp/code
