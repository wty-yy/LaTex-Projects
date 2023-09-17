import time
from tqdm import tqdm

class MyTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        self.sum = 0
        self.max_value = kwargs['total']
        super().__init__(*args, **kwargs)
    
    def update(self, n=1):
        delta = min(self.max_value - self.sum, n)
        super().update(delta)
        self.sum += delta


def test():
    max_value = 100
    with MyTqdm(total=max_value) as pbar:
        sum = 0
        n = 1
        while sum < max_value:
            time.sleep(0.3)
            pbar.update(n)
            sum += n
            n += 1

if __name__ == '__main__':
    test()