import time

class Clock:
    def __init__(self) -> None:
        self.start()

    def start(self):
        self.start_t = time.perf_counter()
    
    def end(self, name=""):
        self.end_t = time.perf_counter()
        print(f"{name} used {self.end_t - self.start_t:.2f} s")