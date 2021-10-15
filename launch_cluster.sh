
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

for sparse in 1; do
  for seed in 1234 4321 3333; do
    for explo_it in 1 10 50; do
    #  sbatch --export=EXPLO_IT=$explo_it,L2=0,INNER=1,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
    #  sleep 1
    #  sbatch --export=EXPLO_IT=$explo_it,L2=1,INNER=1,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
    #  sleep 1
    #  sbatch --export=EXPLO_IT=$explo_it,L2=0,INNER=0,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
    #  sleep 1
    #  sbatch --export=EXPLO_IT=$explo_it,L2=1,INNER=0,DEC_E=0,DEC_OPT=0 cluster_code.sbatch
    #  sleep 1
    #  sbatch --export=EXPLO_IT=$explo_it,L2=0,INNER=1,DEC_E=1,DEC_OPT=0 cluster_code.sbatch
    #  sleep 1
#      sbatch --export=EXPLO_IT=$explo_it,EXPLO_LOSS=$expl,SPARSE=$sparse,DEC_OPT=1 cluster_code.sbatch
#      sleep 1
      sbatch --export=EXPLO_IT=$explo_it,SEED=$seed,SPARSE=$sparse,DEC_OPT=0 cluster_code.sbatch
      sleep 1
    done
  done
done

