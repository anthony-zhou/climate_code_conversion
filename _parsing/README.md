
## Overview

At a high level, the goal of this parsing module is to take in a Fortran codebase and output a DAG containing the relationships between every symbol.

Using that DAG, we want to generate an ordered list of subproblems for input to ChatGPT.

## Details

When it comes to parsing, we have two options implemented:

- fparser
- LSP

`fparser` only works for single files, but not for imported modules.

`LSP` works for imported modules, but its parsing is naive (so it's apt to miss symbols) and it doesn't know how folders might get linked together during compilation. 

Moving forward, enhancing the `fparser` approach with some functionality looking for modules seems like the best option. I only implemented the `LSP` version because it's more general -- every language implementing the language server protocol would be able to use this tool. 

For now, feel free to try out both and make improvements. 

## Preprocessing

The `preprocess.sh` script preprocesses the Fortran scripts using `gcc`, which is necessary for proper usage with `fparser`. 