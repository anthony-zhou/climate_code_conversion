#!/bin/bash

# Usage: ./preprocess.sh <file.f90>

sed -i 's/^[ \t]*#include/#include/' $1
gfortran -cpp -P -E $1 > preprocessed.f90 && mv preprocessed.f90 $1