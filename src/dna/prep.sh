#!/usr/bin/env bash

cat original.csv |
  sed --expression='s/A/0,/g' |
  sed --expression='s/T/1,/g' |
  sed --expression='s/G/2,/g' |
  sed --expression='s/C/3,/g' |
  sed --expression='s/,$//g' > dna.csv

head -800 dna.csv > train.csv
tail -200 dna.csv > test.csv
