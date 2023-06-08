#!/bin/bash
# Should be something like this, to preprocess a file using gfortran

sed -i 's/^[ \t]*#include/#include/' DaylengthMod.f90
gfortran -cpp -P -E DaylengthMod.f90 > preprocessed.f90 && mv preprocessed.f90 DaylengthMod.f90