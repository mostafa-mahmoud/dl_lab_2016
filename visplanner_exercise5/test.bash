#/usr/bin/env bash

for i in {1..20}; do
  echo $i
  rm network* *.csv *.pyc
  python get_data.py
  python train_agent.py
  for j in {1..10}; do
    echo -e "run $i, test $j\n" >> final_results
    python test_agent.py >> final_results
  done
  cat final_results
done
