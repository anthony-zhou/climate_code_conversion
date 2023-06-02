# Climate Code Conversion

Converting Fortran code in climate models to Python code using ChatGPT. 

## Dependencies

Assuming you are running Ubuntu:

1. Install Python 3. 
2. `sudo apt install gfortran cmake` as build tools for Fortran.
3. `sudo apt install m4` as a prerequisite for pFUnit. 
4. Install `pFUnit` (see below)

Once you've installed `pFUnit`, get the absolute path to its installation directory and add it to as an environment variable (e.g. `export PFUNIT_DIR=/home/ubuntu/pFUnit/build/installed`). 

### Installing `pFUnit`

`pFUnit` is a testing library for Fortran that is used by CTSM for unit testing. Here's how you install it:

1. `export FC=gfortran`
2. `git clone https://github.com/Goddard-Fortran-Ecosystem/pFUnit.git`
3. `cd pFUnit`
4. `mkdir build`
5. `cd build`
6. `cmake ..`
7. `make`
8. `make install`

Then pFUnit should be installed. Note that you may get the error (when running `make`) that you don't have `python` installed. If so, you should install it so that `python -V` works from the command line. 

By default, pFUnit should install to `./pfUnit/build/installed`. Go to this directory and `pwd` for the absolute path. 


## Running the tests

Each project in `examples` has a `fortran` folder and a corresponding `python` folder. 

To run the Fortran tests:
1. `cd fortran && ./run_tests.sh`

To run the Python tests:
1. `cd python && ./run_tests.sh`

Tests can currently only be run from the same directory. This design is subject to change. 

## Next Up

We want to try translating some more complicated functions and save the examples here. Once we move to even bigger programs, we'll need to do some automated chunking to split up files into reasonable chunks. 

Currently we can just test using the ChatGPT UI. 