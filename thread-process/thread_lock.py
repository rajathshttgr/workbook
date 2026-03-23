# race condition

import threading
import time

lock = threading.Lock()

counter = 0


def increment():
    global counter
    for _ in range(1000):
        with lock:  # try without lock to learn race condition
            temp = counter
            time.sleep(0.0001)  # force context switch
            temp += 1
            counter = temp


def main():
    threads = []

    for _ in range(10):
        t = threading.Thread(target=increment)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Expected:", 10000)
    print("Actual:", counter)


if __name__ == "__main__":
    main()


"""
# Acquire Lock
lock.acquire()
# release lock
lock.release()

# automatic lock release
with lock:
    # critical section
"""
