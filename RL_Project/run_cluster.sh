
#! /bin/bash

declare -a StringArray=("dqn" "reinforce" "ppo")

for a in "${StringArray[@]}"; do
  for mbs in 1 10 100; do
    sbatch --export=A=$a,MBS=$mbs sbatch_cluster.sbatch
    sleep 1
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

