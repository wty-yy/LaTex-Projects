import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class SEvaluate():
    
    def __init__(self):
        self.path = 'Result/'+'/'+'exp'
        self.best = 0x3f3f3f3f
        i = 0
        while True:
            folder = os.path.exists(self.path + str(i))
            if folder is False:
                self.path = self.path + str(i) + '/'
                os.makedirs(self.path)
                break
            i += 1
            
    def visualize(self, train, test, vis_type, model):
        if self.best > test[len(test)-1] and vis_type=='loss':
            self.best = test[len(test)-1]
            torch.save(model, self.path+'model.pkl')

        index = np.arange(len(train))
        index_ = np.arange(len(test))*10
        
        plt.figure(1)
        plt.grid(color='#7d7f7c', linestyle='-.')
        plt.plot(index, train, 'r', linewidth=1.5, label="train")
        plt.plot(index_, test, '2c--', linewidth=1.5, label="test")
        plt.title(vis_type)
        plt.xlabel('epoch')
        plt.ylabel(vis_type)
        plt.legend(loc=1)
        plt.savefig(self.path + vis_type+'.jpg', dpi=300)
        plt.clf()      