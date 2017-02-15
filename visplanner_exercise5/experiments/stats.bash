#!/usr/bin/env bash

for experiment in *.txt;
do 
    echo $experiment ;
    cat $experiment | grep ^[01]. | python -c "import sys, numpy; acc_lists = zip(*map(lambda x: map(float, x.split(' ')), sys.stdin.read().split('\n')[:-1])); print(str.join('\n', (('%d tests, mean: %.5f, min: %.5f, max: %.5f, std: %.5f' % (numpy.size(acc), numpy.mean(acc), numpy.min(acc), numpy.max(acc), numpy.std(acc))) for acc in acc_lists)));" ;
done

