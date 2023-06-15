#!/bin/bash
# Should be something like this, to preprocess a file using gfortran

sed -i 's/^[ \t]*#include/#include/' $1
gfortran -cpp -P -E $1 > preprocessed.f90 && mv preprocessed.f90 $1