#!/bin/bash

# Download data for MEG analysis
# 
# This script is used to download the necessary data for the MEG source reconstruction from the 
# Amazon S3 storage of the Human Connectome Project data.

set -x
DATAPATH=/data/gpfs-1/users/kollerd_c/work/travelingwaves/data/hcp_meg
subjectlist=$DATAPATH/subjects.txt

while read -r subject;
do
    mkdir -p $DATAPATH/$subject
    mkdir -p $DATAPATH/$subject/MEG/anatomy
    mkdir -p $DATAPATH/$subject/MEG/Restin/baddata
    mkdir -p $DATAPATH/$subject/MEG/Restin/icaclass
    mkdir -p $DATAPATH/$subject/MEG/Restin/baddata
    mkdir -p $DATAPATH/$subject/T1w/$subject
    mkdir -p $DATAPATH/$subject/unprocessed/MEG/1-Rnoise/4D
    mkdir -p $DATAPATH/$subject/unprocessed/MEG/3-Restin/4D

    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/MEG/anatomy \
        $DATAPATH/$subject/MEG/anatomy \
        --recursive \
        --region us-east-1

    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/MEG/Restin/baddata \
        $DATAPATH/$subject/MEG/Restin/baddata \
        --recursive \
        --region us-east-1

    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/MEG/Restin/icaclass \
        $DATAPATH/$subject/MEG/Restin/icaclass \
        --recursive \
        --region us-east-1

    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/T1w/$subject \
        $DATAPATH/$subject/T1w/$subject \
        --recursive \
        --region us-east-1

    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/MEG/1-Rnoise/4D \
        $DATAPATH/$subject/unprocessed/MEG/1-Rnoise/4D \
        --recursive \
        --region us-east-1

    aws s3 cp \
        s3://hcp-openaccess/HCP_1200/$subject/unprocessed/MEG/3-Restin/4D \
        $DATAPATH/$subject/unprocessed/MEG/3-Restin/4D \
        --recursive \
        --region us-east-1

done < $subjectlist
set +x