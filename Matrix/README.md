# MPI Matrix Multiplication -- README

This project demonstrates distributed matrix multiplication using
**mpi4py** in Python.\
The program splits matrix **A** across multiple MPI processes,
broadcasts matrix **B**, performs local multiplication, and gathers the
results into the final matrix **C**.

------------------------------------------------------------------------

## 📌 Features

-   Uses **MPI Scatterv** and **Gatherv** for balanced row distribution.
-   Works with any number of processes.
-   Ensures correctness by verifying `C == A.dot(B)`.
-   Optionally prints matrices when size ≤ 20.

------------------------------------------------------------------------

## 🛠 Requirements

-   Python 3
-   NumPy\
-   mpi4py\
-   An MPI implementation (OpenMPI / MPICH)

Install dependencies:

``` bash
pip install numpy mpi4py
```

------------------------------------------------------------------------

## ▶️ How to Run

### **1. Run with default size (200×200 matrix):**

``` bash
mpirun -np 4 python3 program.py
```

### **2. Run with custom size:**

``` bash
mpirun -np 4 python3 program.py --n 500
```

------------------------------------------------------------------------

## 📂 Program Flow Summary

1.  **Process 0** generates matrices `A` and `B`.

2.  Matrix **B** is broadcast to all processes.

3.  Matrix **A** is divided row-wise using:

    -   `Scatterv` for uneven row distribution.

4.  Each process computes its portion of `C`:

    ``` python
    local_C = local_A.dot(B)
    ```

5.  Results are gathered with `Gatherv`.

6.  Process 0 verifies and optionally prints matrices.

------------------------------------------------------------------------

## ✔ Verification

The program verifies distributed computation using:

``` python
np.allclose(A.dot(B), C)
```

If mismatch occurs, the program raises an error.

------------------------------------------------------------------------

## 🧪 Example

To test with small matrices:

``` bash
mpirun -np 3 python3 program.py --n 5
```

This will print matrices A, B, and C.

------------------------------------------------------------------------

## 📜 License

This project is free to use for academic and learning purposes.
