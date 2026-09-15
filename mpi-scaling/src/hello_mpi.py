from mpi4py import MPI

comm = MPI.COMM_WORLD      # the group of all processes we launched together
rank = comm.Get_rank()     # this process's unique ID
size = comm.Get_size()     # total number of processes

print(f"Hello from rank {rank} out of {size} processes")
