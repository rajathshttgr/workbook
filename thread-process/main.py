import multiprocessing
import os


def worker():
    print("worker process ID", os.getpid())


if __name__ == "__main__":

    print("Main process ID:", os.getpid())
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()
