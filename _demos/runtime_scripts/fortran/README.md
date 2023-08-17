## `fortran`

This folder contains the source code for leaf-level photosynthesis in Fortran, along with some corresponding unit tests. 

`PhotosynthesisMod.f90` contains a simplified version of the `ci_func` function for leaf-level photosynthesis from `CTSM`. Adapted from [here](https://github.com/ESCOMP/CTSM/blob/bb2a8d2c0c05ebb5417724f98fb0586dc7584ae6/src/biogeophys/PhotosynthesisMod.F90). External dependencies (like `r8` and `quadratic`) are reimplemented in the module for simplicity. 

`test_photosynthesis.pf` contains some unit tests for the function.  

## Running the unit tests

First, make sure you have pFUnit installed and added to your `PATH` (see `../README.md` and `../set_pfunit_dir` for details). 

Then, run the following commands:
```
chmod +x run_tests.sh
./run_tests.sh
```


