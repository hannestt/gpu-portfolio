from mpi4py import MPI
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TOTAL_SAMPLES = 100_000_000
samples_per_process = TOTAL_SAMPLES // size

# Each process independently generates its own random points and counts hits
random.seed(rank)  # different seed per process, so they're not all doing identical work
start = time.time()

count_inside = 0
for _ in range(samples_per_process):
    x = random.random()
    y = random.random()
    if x*x + y*y <= 1.0:
        count_inside += 1

local_time = time.time() - start

# Combine everyone's counts into rank 0
total_inside = comm.reduce(count_inside, op=MPI.SUM, root=0)

# Also find the SLOWEST process's time, since that determines overall wall-clock time
max_time = comm.reduce(local_time, op=MPI.MAX, root=0)

if rank == 0:
    pi_estimate = 4.0 * total_inside / TOTAL_SAMPLES
    print(f"Processes: {size}")
    print(f"Pi estimate: {pi_estimate}")
    print(f"Max process time: {max_time:.4f}s")
