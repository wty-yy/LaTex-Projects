# coding:utf-8
import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def get_time(self):
        return time.time() - self.start

    def show(self):
        print('用时: %.4f s' % (time.time() - self.start))
