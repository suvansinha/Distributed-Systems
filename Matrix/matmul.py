#!/usr/bin/env python3
import argparse
import numpy as np
import time

def mpi_mode(N):
    try:
        from mpi4py import MPI
    except Exception as e:
        raise RuntimeError("mpi4py is required for MPI mode: pip install mpi4py") from e

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

   
    if size < 1:
        if rank == 0:
            print("Error: MPI reported size < 1")
        return None

    
    A = None
    B = None
    seed = 42
    if rank == 0:
        rng = np.random.default_rng(seed)
        A = rng.integers(0, 10, size=(N, N)).astype(np.float64)
        B = rng.integers(0, 10, size=(N, N)).astype(np.float64)

    if rank == 0:
        B_buf = B
    else:
        B_buf = np.empty((N, N), dtype=np.float64)
   
    comm.Bcast(B_buf, root=0)


    base = N // size
    extra = N % size
    rows_per_proc = [base + (1 if i < extra else 0) for i in range(size)]
    displs = [sum(rows_per_proc[:i]) for i in range(size)]
    my_rows = rows_per_proc[rank]


    local_A = np.empty((my_rows, N), dtype=np.float64)


    sendcounts = [r * N for r in rows_per_proc]
    displs_elems = [d * N for d in displs]

   
    if rank == 0:
       
        sendbuf = A.ravel()
        comm.Scatterv([sendbuf, sendcounts, displs_elems, MPI.DOUBLE], local_A, root=0)
    else:
        comm.Scatterv([None, sendcounts, displs_elems, MPI.DOUBLE], local_A, root=0)

    
    comm.Barrier()
    t0 = time.time()

   
    if my_rows > 0:
        local_C = local_A.dot(B_buf)    
    else:
        local_C = np.empty((0, N), dtype=np.float64)

    
    if rank == 0:
        C = np.empty((N, N), dtype=np.float64)
    else:
        C = None

    
    if rank == 0:
       
        comm.Gatherv(local_C, [C, sendcounts, displs_elems, MPI.DOUBLE], root=0)
    else:
        comm.Gatherv(local_C, [None, sendcounts, displs_elems, MPI.DOUBLE], root=0)

    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        elapsed = t1 - t0
        print(f"MPI: N={N}, processes={size}, time={elapsed:.6f} sec\n")

        
        if N <= 20:
            np.set_printoptions(precision=0, suppress=True)
            print("Matrix A:")
            print(A)
            print("\nMatrix B:")
            print(B_buf)
            print("\nResultant Matrix C = A x B:")
            print(C)
            print()

       
        expected = A.dot(B_buf)
        if not np.allclose(expected, C):
           
            print("ERROR: Matrix multiplication result mismatch!")
            if N <= 20:
                print("\nExpected (A.dot(B)):")
                print(expected)
            raise AssertionError("Matrix multiplication result mismatch!")
        else:
            print("Verification: OK (distributed result matches A.dot(B))")

        return C
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI matrix multiplication (fixed scatter/gather)')
    parser.add_argument('--n', type=int, default=200, help='matrix dimension (N x N)')
    args = parser.parse_args()

   
    result = mpi_mode(args.n)
