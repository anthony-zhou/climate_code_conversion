Parse a Fortran file using fparser. 

## Usage

```
python visualize.py
```

## Features

- Generate a DAG for a given Fortran source code file

## Limitations

- `fparser` can't tell the difference between function calls and array references. 
- `fparser` is not built to link modules together, so it only works for single files. 