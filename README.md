

# Sylvester equation solver

## About

Sylvester equation solver: Bartels–Stewart algorithm implementation with a recursive blocked algorithm to solve the triangular system. 


The solver of the triangular system is based on the paper [Recursive blocked algorithms for solving triangular systems—Part I: one-sided and coupled Sylvester-type matrix equations](https://dl.acm.org/doi/10.1145/592843.592845),
*Isak Jonsson, Bo Kågström*, 2002.

This repository stores the code used for a project in a Computational Linear Algebra course ([MATH-453](https://edu.epfl.ch/coursebook/en/computational-linear-algebra-MATH-453)).

The performance benchmark is performed in notebooks, results are presented in the `report.pdf`.

## Project structure

### Algorithm implementation

The core of this project is the function `rtrgsyl(..., std_solver)` in the `recursive.py` file, this implements the recursive blocked 
algorithm of the aforementioned paper. Note that it relies on another solver (`std_solver`) when small matrix sizes are reached.

The interface function `solve_bartel_stewart` in `utils.py` is a wrapper which performs the Schur decomposition, provides
matrices in Schur form to the wrapped solver to solve the triangular system, and maps 
the solution back to the original coordinate system. 

### Benchmarking

The benchmarks are performed in notebooks, via the functions in `bechmark_utils.py`.
