
#! /bin/bash

for ea in 1.0 0.2; do
  sbatch --export=PL=4,BT=10,EM=100,AT=256,ABT=1.,EA=$ea,MGD=4,FT=0 cluster_code.sbatch
  sleep 1
done

for abt in 0.1 0.05; do
  for ea in 1.0 0.2; do
    for ft in 0 1 2 3 4 5; do
      sbatch --export=PL=4,BT=10,EM=100,AT=256,ABT=$abt,EA=$ea,MGD=4,FT=$ft cluster_code.sbatch
      sleep 1
    done
  done
done
