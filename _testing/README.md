How do we generate good unit tests for a given chunk of Fortran source code?

Our proposed approach is as follows:
1. Run the original Fortran module, logging function calls and corresponding arguments.
2. Pass the logged arguments into the GPT-4 API and ask it to generate Fortran unit tests using these test cases. 
3. Generate Python unit tests from the Fortran unit tests
4. Generate Python code to pass the Python unit tests
5. Iterate on 4 until the Python outputs match the Fortran outputs.

## What didn't work

- Pynguin for generating automatic unit tests -- tests were bad (see `./pynguin` for an example)

# Examples of testing in Fortran and Python

Under `examples/` we have some GPT-translated Python unit tests and their corresponding Fortran sources. They're notably simple functions where ChatGPT can come up with its own test cases -- more complex functions might need us to log function arguments and generate realistic unit tests, or take some kind of evolutionary approach to maximize test coverage. 

Here we document a process for running these example tests (assuming one day we can generate such tests automatically in the aforementioned workflow).

## Dependencies

1. `sudo apt install gfortran cmake` as build tools for Fortran.
2. `sudo apt install m4` as a prerequisite for pFUnit. 
3. Install `pFUnit` (see below)

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


## Testing

Use `pytest` for Python unit tests.