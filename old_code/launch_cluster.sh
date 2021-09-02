
#! /bin/bash

for pl in 2 3; do
  for at in 20 100; do
    for ea in 1.0 0.5 0.0; do
      sbatch --export=PL=$pl,AT=$at,EA=$ea cluster_code.sbatch
      sleep 1
    done
  done
done

#sbatch --export=BT=100,MT=0,ACT='tanh',ADAM=1e-5,NORMA=1,GRDCLIP=1 cluster_code.sbatch
#sleep 1
#sbatch --export=BT=10,MT=0,ACT='tanh',ADAM=1e-5,NORMA=1,GRDCLIP=1 cluster_code.sbatch
#sleep 1
#sbatch --export=BT=100,MT=1,ACT='tanh',ADAM=1e-5,NORMA=1,GRDCLIP=1 cluster_code.sbatch
#sleep 1
#sbatch --export=BT=100,MT=0,ACT='relu',ADAM=1e-5,NORMA=1,GRDCLIP=1 cluster_code.sbatch
#sleep 1
#sbatch --export=BT=100,MT=0,ACT='tanh',ADAM=1e-8,NORMA=1,GRDCLIP=1 cluster_code.sbatch
#sleep 1
#sbatch --export=BT=100,MT=0,ACT='tanh',ADAM=1e-5,NORMA=0,GRDCLIP=1 cluster_code.sbatch
#sleep 1
#sbatch --export=BT=100,MT=0,ACT='tanh',ADAM=1e-5,NORMA=1,GRDCLIP=0 cluster_code.sbatch
#sleep 1

