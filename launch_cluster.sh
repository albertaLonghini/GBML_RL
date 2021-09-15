
#! /bin/bash

#for sparse in 0 1; do
#  for l2 in 0 1; do
#    for inner in 0 1; do
#      for decouple in 0 1; do
#        sbatch --export=SPARSE=$sparse,L2=$l2,INNER=$inner,DECOUPLE=$decouple cluster_code.sbatch
#        sleep 1
#      done
#    done
#  done
#done

for sparse in 0 1; do
  sbatch --export=SPARSE=$sparse,L2=0,INNER=1,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
  sleep 1
  sbatch --export=SPARSE=$sparse,L2=1,INNER=1,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
  sleep 1
  sbatch --export=SPARSE=$sparse,L2=0,INNER=0,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
  sleep 1
  sbatch --export=SPARSE=$sparse,L2=1,INNER=0,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
  sleep 1
  sbatch --export=SPARSE=$sparse,L2=0,INNER=1,DEC_E=1,DEC_OPT=0 cluster_code.sbatch
  sleep 1
  sbatch --export=SPARSE=$sparse,L2=1,INNER=1,DEC_E=1,DEC_OPT=0 cluster_code.sbatch
  sleep 1
  sbatch --export=SPARSE=$sparse,L2=1,INNER=1,DEC_E=1,DEC_OPT=1 cluster_code.sbatch
  sleep 1
done

