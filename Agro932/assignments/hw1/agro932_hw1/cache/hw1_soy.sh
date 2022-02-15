#!/bin/bash -l
#SBATCH -D /common/agrobinf/masonlien/courses/2022-agro932-lab/data/hw1
#SBATCH -o /common/agrobinf/masonlien/courses/2022-agro932-lab/data/hw1/log/hw1_soy_out_%j.txt
#SBATCH -e /common/agrobinf/masonlien/courses/2022-agro932-lab/data/hw1/log/hw1_soy_err_%j.txt
#SBATCH -J hw1_soy
#SBATCH -t 24:00:00

set -e
set -u

cd soybean/

wgsim Glycine_max.Glycine_max_v2.1.dna.chromosome.1.fa -N 5000 -1 100 -2 100 -r 0.01 -e 0 \
-R 0 -X 0 -S 1234567 l1.R1.fq l1.R2.fq

module load bwa samtools bcftools

bwa index Glycine_max.Glycine_max_v2.1.dna.chromosome.1.fa


# alignment
for i in {1..20}; do bwa mem Glycine_max.Glycine_max_v2.1.dna.chromosome.1.fa l$i.read1.fq l$i.read2.fq | samtools view -bSh - > l$i.bam; done
# sort
for i in *.bam; do samtools sort $i -o sorted_$i; done
# index them
for i in sorted*.bam; do samtools index $i; done

### check mapping statistics
samtools flagstat sorted_l1.bam

### index the genome assembly
samtools faidx Glycine_max.Glycine_max_v2.1.dna.chromosome.1.fa

### run 'mpileup' to generate VCF format
ls sorted_l*bam > bamlist.txt
samtools mpileup -g -f Glycine_max.Glycine_max_v2.1.dna.chromosome.1.fa -b bamlist.txt > myraw.bcf
bcftools call myraw.bcf -cv -Ob -o snps.bcf

### Extract allele frequency at each position
bcftools query -f '%CHROM %POS %AF1\n' snps.bcf > frq.txt
bcftools query -f '%CHROM %POS %REF %ALT [\t%GT]\n' snps.bcf > geno.txt

