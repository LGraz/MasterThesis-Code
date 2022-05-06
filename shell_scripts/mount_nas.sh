#!/usr/bin/bash
sudo mount -t cifs -o user=lgraz@ethz.ch,uid=1000,gid=12289 //hest.nas.ethz.ch/green_groups_kp_public $M
