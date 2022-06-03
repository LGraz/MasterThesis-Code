#!/usr/bin/bash
M=/home/lukas/Documents/ETH/MASTER_THESIS

#copy files from remote
scp -rC lgraz@sftpmath.math.ethz.ch:/home/computation_results $M/code/data
