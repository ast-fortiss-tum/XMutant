#!/bin/bash
python -m black . --line-length 100 --preview

arr=("XMutant-MNIST") # "XMutant-IMDB" "XMutant-LK-ADS" "XMutant-MNIST"
for elem in "${arr[@]}"
do
  darglint -s sphinx "${elem}/."
  pyflakes "${elem}/."
  isort --profile black "${elem}/."
done
