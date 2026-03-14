# thread

import threading
import time
import sys


def worker(name, len):
    for i in range(len):
        print(f"Thread {name}: {i}")
        time.sleep(1)


def main():
    t1 = threading.Thread(target=worker, args=("a", 2))
    t2 = threading.Thread(target=worker, args=("b", 15))
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("process completed")
    return "success"


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("please enter 2 arguments")
    else:
        result = main()
        print(result)
