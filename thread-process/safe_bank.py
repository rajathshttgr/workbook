# Safe Bank Account

import threading


class BankAccount:
    def __init__(self, balance):
        self.balance = balance
        self.lock = threading.Lock()

    def withdraw(self, amount):
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount


account = BankAccount(1000)


def withdraw_task():
    for _ in range(100):
        account.withdraw(5)


t1 = threading.Thread(target=withdraw_task)
t2 = threading.Thread(target=withdraw_task)

t1.start()
t2.start()

t1.join()
t2.join()

print("Final Balance:", account.balance)
