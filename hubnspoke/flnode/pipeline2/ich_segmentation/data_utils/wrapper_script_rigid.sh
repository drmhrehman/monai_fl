#!/bin/bash
# Wrapper script for SPM clinical preprocessing
# Takes two arguments:
# 1: input image
# 2: output directory
# Input image will be copied to output directory before preprocessing

paths="{{'$1'},{'$2'}}"

outputDir=$3



opt="struct('prefix','p',\
'dir_out','${outputDir}',\
'do', struct(\
'res_orig',true,\
'real_mni',true,\
'nm_reorient',true,\
'vx',true,\
'reslice',true,\
'bb_spm',true,\
'crop',true),\
'vx',struct(\
'deg',1,\
'size',1),\
'bb',struct(\
'dim',[256 256 256]),\
'realign2mni',struct(\
'rigid',true))"


echo Preprocessing image:
echo $paths
echo Using output directory:
echo $outputDir
echo Using options:
echo $opt
echo

/opt/spm12/spm12 eval "RunPreproc($paths,$opt)"
