# Climate Code Conversion

Converting Fortran code in climate models to Python code using ChatGPT. 

## Dependencies

Assuming you are running Ubuntu:

1. Install Python 3. 
2. `sudo apt install gfortran cmake` as build tools for Fortran.
3. `sudo apt install m4` as a prerequisite for pFUnit. 
4. Install `pFUnit` (see below)
5. `pip install -r requirements.txt` to install all Python packages
6. `pip install -e .` to  add the local `translation` package to your path (needed for absolute imports)

Once you've installed `pFUnit`, get the absolute path to its installation directory and add it to as an environment variable (e.g. `export PFUNIT_DIR=/home/ubuntu/pFUnit/build/installed`). 

## Environment Variables

To run the LLM code, you need an `OPENAI_API_KEY` environment variable. You can set this from a `.env` file in this directory. For example: `touch .env && echo "OPENAI_API_KEY=sk-yadayadayada" > .env`.

## Folder Structure

`archive` contains old files.
`fortran` contains a Fortran version of photosynthesis.
`python_ci_func` contains a Python version of photosynthesis.
`tests` contains unit tests for the `translation` module.
`translation` contains the `translation` module, which is primarily a CLI used to translate code automatically. 

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

Python unit tests can be run with `pytest`. 

## Next Up

We want to try translating some more complicated functions and save the examples here. Once we move to even bigger programs, we'll need to do some automated chunking to split up files into reasonable chunks. 

Currently we can just test using the ChatGPT UI. 