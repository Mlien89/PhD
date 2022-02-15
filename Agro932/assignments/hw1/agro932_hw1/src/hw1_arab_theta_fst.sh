#!/bin/sh
#SBATCH -D /common/agrobinf/masonlien/PhD/Agro932/assignments/hw1/agro932_hw1/
#SBATCH -o /common/agrobinf/masonlien/PhD/Agro932/assignments/hw1/agro932_hw1/logs/hw1/hw1_arab_theta_fst_out.txt
#SBATCH -e /common/agrobinf/masonlien/PhD/Agro932/assignments/hw1/agro932_hw1/logs/hw1/hw1_arab_theta_fts_err.txt
#SBATCH -J hw1_arab_theta_fst
#SBATCH -t 24:00:00

set -e
set -u

cd data/

module load R

geno <- read.table("geno.txt", header=FALSE)
names(geno) <- c("chr", "pos", "ref", "alt", "l1", "l2", "l3", "l4", "l5")
head(geno)
#for(i in 5:9){
  # replace slash and everything after it as nothing
#  geno$newcol <- gsub("/.*", "", geno[,i] )
  # extract the line name
#  nm <- names(geno)[i]
  # assign name for this allele
#  names(geno)[ncol(geno)] <- paste0(nm, sep="_a1")
#  geno$newcol <- gsub(".*/", "", geno[,i] )
#  names(geno)[ncol(geno)] <- paste0(nm, sep="_a2")
#}


