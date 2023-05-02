from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def my_func(i):
    print(f"Executing my_func pid={os.getpid()} with argument={i}")
    return i * 2

if __name__ == "__main__":
    jobs = [1, 2, 3, 4]

    with ProcessPoolExecutor(max_workers=4) as pool:
        results = [pool.submit(my_func, j) for j in jobs]
        
        for f in as_completed(results):
            print(f"result={f.result()}", flush=True)