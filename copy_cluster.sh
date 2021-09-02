
scp -P 2222 -i ~/cluster_key ./*.py longhini@moria.csc.kth.se:~/GBML_RL
sleep 1
scp -P 2222 -i ~/cluster_key ./*.s* longhini@moria.csc.kth.se:~/GBML_RL
sleep 1
scp -P 2222 -i ~/cluster_key -r ./agents longhini@moria.csc.kth.se:~/GBML_RL
sleep 1
scp -P 2222 -i ~/cluster_key -r ./runners longhini@moria.csc.kth.se:~/GBML_RL
sleep 1
