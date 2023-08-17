# Code Translation CLI

Tool for iterative translation from Fortran to Python, using the GPT-4 API. 

![Demo of code translation CLI](demos/demo.svg)

## Try it out

Make a virtualenv and then `pip install -r requirements.txt`.

Create a `.env` file containing an OpenAI API key with access to GPT-4. Optionally include a PromptLayer API key for logging API requests.  

Select or create an input file with some Fortran code. Then run `python main.py --infile <path_to_input_file> --outfile <path_to_output_file> --testfile <path_to_output_test_file>`. 

Some notes: 
- Make sure your `infile` contains valid Fortran code, ideally just a single function. 
- If you run `python main.py` by itself the tool defaults to the demo (sum two numbers). 
- GPT-4 is quite slow. 

## How it works

Start with a piece of code to translate. Ideally this is just a single function, subroutine, or type. Because then we can run unit tests on it. 

Then we generate Python source code, and generate Python unit tests, using prompts to GPT-4. 

Observe that we currently don't generate Fortran unit tests. This is weird, because then we can't make sure that the Python behavior matches up to the Fortran behavior. 

Note that this module knows nothing about the filesystem or dependency graph -- all it has is the chunk of source code that it's been given. 

Finally, we run the code in a Docker image and pass unit test outputs back into ChatGPT for updating. 

## Solving problems by reading unit test output

Sometimes GPT-4 makes a mistake on its first go, and it needs to solve some failing unit tests. Here's an example:

![Demo where GPT-4 solves its own problems](demos/ci_func_demo.svg)

Video version of this demo: https://asciinema.org/a/603256

Note here that GPT-4 solves its problems by changing the unit tests to match the code output.  

This is kind of cheating -- it's like changing the answer key to match what you answered on a test. 

But in fact the unit tests show that the generated code is working, albeit with less test coverage than we would want. 

## Limitations

- Currently Docker images for testing are loaded statically with some default dependencies (`numpy` and `pytest`). Your code may rely on other packages. 
  - Solution is eventually to dynamically generate a sandbox environment based on the generated code -- let ChatGPT come up with its own sandbox, as in https://github.com/modal-labs/devlooper
- Very slow for any sizable functions. The ci_func demo is 4x sped up -- the original run took 5 minutes from start to finish. 
