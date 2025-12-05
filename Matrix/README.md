# Matrix Multiplication Program (Matmul.py)

This program performs matrix multiplication using either regular NumPy
execution or MPI-based parallel execution. It is designed for
benchmarking and experimenting with distributed computation.

## Features

-   Generate two random N×N matrices.
-   Multiply matrices using:
    -   **Normal mode** (NumPy)
    -   **MPI mode** (mpi4py)
-   Measure execution time.
-   Command-line arguments for easy control.

## Usage

### Normal Mode

``` bash
python3 Matmul.py --mode normal --n 500
```

### MPI Mode

``` bash
mpiexec -n 4 python3 Matmul.py --mode mpi --n 500
```

## Arguments

  Argument   Description
  ---------- ------------------------------------
  `--mode`   Execution mode (`normal` or `mpi`)
  `--n`      Size of the matrix (N×N)

## Requirements

-   Python 3
-   NumPy
-   mpi4py (only for MPI mode)
-   An MPI implementation (e.g., OpenMPI)

## Output Example

    Normal: N=500, procs=1, time=0.0123s

or

    MPI: N=500, procs=4, time=0.0056s

## Author

Generated README based on your provided script.
