import torch
import torch.nn as nn
import torch.nn.functional as fun


class ChannelAttentionMoudle(nn.Module):
    
    def __init__(self, channel, r):
        super(ChannelAttentionMoudle, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(channel, channel//r), nn.Linear(channel//r, channel))
        
    def forward(self, feature):
        B, C, H, W = feature.shape
        f_max = self.MLP(fun.max_pool2d(feature, (H, W)).squeeze(dim=-1).squeeze(dim=-1))
        f_avg = self.MLP(fun.avg_pool2d(feature, (H, W)).squeeze(dim=-1).squeeze(dim=-1))
        f = torch.sigmoid(f_max + f_avg).unsqueeze(dim=-1).unsqueeze(dim=-1)
        refine_feature = feature*f
        return refine_feature
    
class SpatialAttentionMoudle(nn.Module):
    
    def __init__(self):
        super(SpatialAttentionMoudle, self).__init__()
        self.CONV = nn.Conv2d(2, 1, (7,7), 1, 3)
        
    def forward(self, feature):
        B, C, H, W = feature.shape
        f_max, _ = torch.max(feature, dim=1)
        f_avg = torch.sum(feature, dim=1) / C
        f = self.CONV(torch.stack([f_max, f_avg], dim=1))
        refine_feature = feature*f
        return refine_feature

class CBAM(nn.Module):
    
    def __init__(self, channel, r):
        super(CBAM, self).__init__()
        self.CAM = ChannelAttentionMoudle(channel, r)
        self.SAM = SpatialAttentionMoudle()
        
    def forward(self, feature):
        refine_1_feature = self.CAM(feature)
        refine_2_feature = self.CAM(refine_1_feature)
        return refine_2_feature

class VGG(nn.Module):
    
    def __init__(self, channel, num_class):
        super(VGG, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(channel, 64, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(64, 64, (3,3), 1, 1), nn.ReLU(),
                                   nn.MaxPool2d((2, 2)))
        self.Conv2 = nn.Sequential(nn.Conv2d(64, 128, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(128, 128, (3,3), 1, 1), nn.ReLU(),
                                   nn.MaxPool2d((2, 2)))
        self.Conv3 = nn.Sequential(nn.Conv2d(128, 256, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(256, 256, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(256, 256, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(256, 256, (3,3), 1, 1), nn.ReLU(),
                                   nn.MaxPool2d((2, 2)))
        self.Conv4 = nn.Sequential(nn.Conv2d(256, 512, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(),
                                   nn.MaxPool2d((2, 2)))
        self.Conv5 = nn.Sequential(nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), 
                                   nn.MaxPool2d((2, 2)))
    
        self.MLP = nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 1024), nn.ReLU(),
                                 nn.Linear(1024, num_class), nn.Softmax(dim=1))
        

    
    def forward(self, image):
        feature =  self.Conv5(self.Conv4(self.Conv3(self.Conv2(self.Conv1(image)))))
        flat_feature = torch.flatten(feature, start_dim=1)
        probability = self.MLP(flat_feature)
        return probability
    
class VGG_CBAM(nn.Module):
    
    def __init__(self, channel, num_class, r):
        super(VGG_CBAM, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(channel, 64, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(64, 64, (3,3), 1, 1), nn.ReLU(), CBAM(64, r),
                                   nn.MaxPool2d((2, 2)))
        self.Conv2 = nn.Sequential(nn.Conv2d(64, 128, (3,3), 1, 1), nn.ReLU(),
                                   nn.Conv2d(128, 128, (3,3), 1, 1), nn.ReLU(), CBAM(128, r),
                                   nn.MaxPool2d((2, 2)))
        self.Conv3 = nn.Sequential(nn.Conv2d(128, 256, (3,3), 1, 1), 
                                   nn.Conv2d(256, 256, (3,3), 1, 1), nn.ReLU(), CBAM(256, r),
                                   nn.Conv2d(256, 256, (3,3), 1, 1), 
                                   nn.Conv2d(256, 256, (3,3), 1, 1), nn.ReLU(), CBAM(256, r),
                                   nn.MaxPool2d((2, 2)))
        self.Conv4 = nn.Sequential(nn.Conv2d(256, 512, (3,3), 1, 1),  
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(), CBAM(512, r),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.BatchNorm2d(512),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(), CBAM(512, r),
                                   nn.MaxPool2d((2, 2)))
        self.Conv5 = nn.Sequential(nn.Conv2d(512, 512, (3,3), 1, 1), 
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(), CBAM(512, r),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.BatchNorm2d(512),
                                   nn.Conv2d(512, 512, (3,3), 1, 1), nn.ReLU(), CBAM(512, r),
                                   nn.MaxPool2d((2, 2)))
    
        self.MLP = nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 1024), nn.ReLU(),
                                 nn.Linear(1024, num_class))
    
    def forward(self, image):
        feature =  self.Conv5(self.Conv4(self.Conv3(self.Conv2(self.Conv1(image)))))
        flat_feature = torch.flatten(feature, start_dim=1)
        probability = self.MLP(flat_feature)
        return probability
        
if __name__ == '__main__':
    model = VGG_CBAM(3, 256, 8)
    image = torch.ones((32, 3, 224, 224))
    print(model.forward(image).shape)