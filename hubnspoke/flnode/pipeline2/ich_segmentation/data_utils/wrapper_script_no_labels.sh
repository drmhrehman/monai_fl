#!/bin/bash
# Wrapper script for SPM clinical preprocessing
# Takes two arguments:
# 1: input image
# 2: output directory
# Input image will be copied to output directory before preprocessing

paths="{'$1'}"

outputDir=$2

#opt="struct('prefix','p',\
##'dir_out','${outputDir}',\
##'do', struct(\
##'res_orig',true,\
##'real_mni',true,\
##'nm_reorient',true,\
##'segment',true,\
##'normalise',true),\
##'segment',struct(\
##'write_tc',[true(6,1),false(6,2),true(6,1)],\
##'write_df',[1,1]),\
##'normalise',struct(\
##'vox',[1,1,1],\
##'bb',[-127,-145,-120;128,110,135]))"

opt="struct('prefix','p',\
'dir_out','${outputDir}',\
'do', struct(\
'res_orig',true,\
'real_mni',true,\
'nm_reorient',true,\
'segment',true,\
'crop',true,\
'normalise',true),\
'segment',struct(\
'write_tc',false(6,4),\
'write_bf',false,\
'write_df',false),\
'normalise',struct(\
'vox',[1,1,1],\
'bb',[-127,-145,-120;128,110,135]))"

#opt="struct('prefix','p',\
#'dir_out','${outputDir}',\
#'do', struct(\
#'res_orig',true,\
#'real_mni',true,\
#'nm_reorient',true,\
#'segment',true,\
#'normalise',true),\
#'segment',struct(\
#'write_tc',false,\
#'write_df', false),\
#'normalise',struct(\
#'vox',[1,1,1],\
#'bb',[-127,-145,-120;128,110,135]))"

echo Preprocessing image:
echo $paths
echo Using output directory:
echo $outputDir
echo Using options:
echo $opt
echo

/opt/spm12/spm12 eval "RunPreproc($paths,$opt)"
