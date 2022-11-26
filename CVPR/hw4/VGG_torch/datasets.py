import cv2
import pdb
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(224, 224)),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class MyDataset(Dataset):
    
    def __init__(self, root_path, config_path, device):
        super(MyDataset, self).__init__()
        files = open(config_path, 'r').read().split('\n')
        self.device = device
        self.transform = transform
        self.img_path = []
        self.label = []
        for file in files:
            file = file.split('/')
            file_path = root_path + file[-2] + '/' + file[-1].split(' ')[0]
            class_name = int(file[-1].split(' ')[1].split('.')[0])
            self.img_path.append(file_path)
            self.label.append(class_name)
        
    def __getitem__(self, index):
        img = cv2.imread(self.img_path[index])
        img = self.transform(img)
        label = torch.Tensor([self.label[index]])-1
        return torch.Tensor(img).float().to(self.device), torch.Tensor(label).long().to(self.device)
    
    def __len__(self):
       return len(self.img_path)
    
    
if __name__ == '__main__':
    root_path = 'data/256_ObjectCategories/'
    config_path = 'train.txt'
    dataset = MyDataset(root_path, config_path, 'cuda:0')
    
   
        
