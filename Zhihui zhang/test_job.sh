#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:0:10
#SBATCH --mem=100M
#SBATCH --account=math027744


echo 'My first job'
hostname