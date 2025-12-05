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

    
    if rank == 0:
        A = np.random.randint(0, 10, size=(N, N)).astype(np.float64)
        B = np.random.randint(0, 10, size=(N, N)).astype(np.float64)
    else:
        A = None
        B = None

    
    if rank == 0:
        B_buf = B
    else:
        B_buf = np.empty((N, N), dtype=np.float64)
    comm.Bcast(B_buf, root=0)

    
    rows_per_proc = [N // size + (1 if i < (N % size) else 0) for i in range(size)]
    displs = [sum(rows_per_proc[:i]) for i in range(size)]
    my_rows = rows_per_proc[rank]

   
    local_A = np.empty((my_rows, N), dtype=np.float64)

    
    sendcounts = [r * N for r in rows_per_proc]
    displs_elems = [d * N for d in displs]

    if rank == 0:
        comm.Scatterv([A.flatten(), sendcounts, displs_elems, MPI.DOUBLE], local_A.flatten(), root=0)
    else:
        comm.Scatterv([None, sendcounts, displs_elems, MPI.DOUBLE], local_A.flatten(), root=0)

    comm.Barrier()
    t0 = time.time()

    
    local_C = local_A.dot(B_buf)

   
    if rank == 0:
        C = np.empty((N, N), dtype=np.float64)
    else:
        C = None

    recvcounts = sendcounts  
    if rank == 0:
        comm.Gatherv(local_C.flatten(), [C.flatten(), recvcounts, displs_elems, MPI.DOUBLE], root=0)
    else:
        comm.Gatherv(local_C.flatten(), [None, recvcounts, displs_elems, MPI.DOUBLE], root=0)

    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        print(f"MPI: N={N}, procs={size}, time={t1 - t0:.4f}s")
        
        if N <= 200:
            A_full = A
            B_full = B
            expected = A_full.dot(B_full)
            assert np.allclose(expected, C), "Result mismatch!"
        return C
    else:
        return None





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Matrix Multiplication example')
    parser.add_argument('--mode', choices=['mpi', 'normal'], default='normal', help='distributed mode')
    parser.add_argument('--n', type=int, default=200, help='matrix dimension (N x N)')
    args = parser.parse_args()

    
        
    result = mpi_mode(args.n)
        
  
