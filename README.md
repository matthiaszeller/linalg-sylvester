

# Sylvester equation solver

## About

Sylvester equation solver: Bartels–Stewart algorithm implementation with a recursive blocked algorithm to solve the triangular system. 


The solver of the triangular system is based on the paper [Recursive blocked algorithms for solving triangular systems—Part I: one-sided and coupled Sylvester-type matrix equations](https://dl.acm.org/doi/10.1145/592843.592845),
*Isak Jonsson, Bo Kågström*, 2002.

This repository stores the code used for a project in a Computational Linear Algebra course ([MATH-453](https://edu.epfl.ch/coursebook/en/computational-linear-algebra-MATH-453)).

The performance benchmark is performed in notebooks, results are presented in the `report.pdf`:

* [1-Benchmark-intro.ipynb](1-Benchmark-intro.ipynb): compares solving times of scipy's solver with a linear system solver,
show time decomposition for the three phases of Bartel-Stewart algorithm
  
* [2-Benchmark-blks.ipynb](2-Benchmark-blks.ipynb): optimal block size selection for `rtrgsyl`

* [3-Benchmark-rtrgsyl.ipynb](3-Benchmark-rtrgsyl.ipynb): compare solving times (phase 2 of Bartel-Stewart) of `rtrgsyl` 
and scipy's solver
  
* [4-Benchmark-scalability.ipynb](4-Benchmark-scalability.ipynb): explore scalability of `rtrgsyl` by varying number of threads

## Quick test

Run the `script.py` file to perform a quick test of the solver, and show the timings of the three phases of 
Bartel-Stewart for `rtrgsyl` (with scipy's solver) vs a linear system solver:

```
$ python3 script.py
Created random matrices with shapes:
(200, 200), (200, 200), (200, 200)

Solving Sylvester equation with Bartels Stewart and rtrgsyl ...
Checking validity of solution by plugging X into equation...
solution is correct

solving times:
i) schur decomp : 	0.0662
ii) rtrgsyl: 		0.0213
iii) map back: 		0.000246

Solving Sylvester equation with Bartels Stewart and solving small systems as linear systems ...
solution is correct

solving times:
i) schur decomp : 	0.0647
ii) rtrgsyl: 		2.6
iii) map back: 		0.000271
```


## Project structure

### Algorithm implementation

The core of this project is the function `rtrgsyl(..., std_solver)` in the `recursive.py` file, this implements the recursive blocked 
algorithm of the aforementioned paper. Note that it relies on another solver (`std_solver`) when small matrix sizes are reached.

The interface function `solve_bartel_stewart` in `utils.py` is a wrapper which performs the Schur decomposition, provides
matrices in Schur form to the wrapped solver to solve the triangular system, and maps 
the solution back to the original coordinate system. 

### Benchmarking

The benchmarks are performed in notebooks, via the functions in `bechmark_utils.py`.

### Unit tests

Unit tests are available in:

* `test_split_matrix.py`
* `test_rtrgsyl.py`
