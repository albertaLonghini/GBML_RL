
#! /bin/bash

for sparse in 0 1; do
  for l2 in 0 1; do
    for inner in 0 1; do
      for decouple in 0 1; do
        sbatch --export=SPARSE=$sparse,L2=$l2,INNER=$inner,DECOUPLE=$decouple cluster_code.sbatch
        sleep 1
      done
    done
  done
done

