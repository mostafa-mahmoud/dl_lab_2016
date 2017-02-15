#/usr/bin/env bash

source ~/.bash_profile
for i in {1..7}; do
  echo $i
  rm network* *.csv *.pyc
  python get_data.py
  python train_agent.py
  for j in {1..10}; do
    echo -e "run $i, test $j\n" >> final_results.txt
    python test_agent.py 1>> final_results.txt
  done
  cat final_results.txt
done
