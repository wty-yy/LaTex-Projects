import os
import os.path as osp
import random

root=r'data/256_ObjectCategories'
data_list=[]
class_list=os.listdir(root)
test_percent=0.2         #设置测试集的占比
for i in range(len(class_list)):
    item_list=os.listdir(osp.join(root,class_list[i]))
    print(len(item_list))
    for j in range(len(item_list)):
        item_path=osp.join(root,class_list[i],item_list[j])
        data_list.append(item_path)
ftest = open(r'256_ObjectCategories_test.txt', 'w')
ftrain = open(r'256_ObjectCategories_train.txt', 'w')

num = len(data_list)
print(num)
print(num)
tv = int(num * test_percent)
testval=random.sample(data_list, tv)


for i in range(len(data_list)):
    if data_list[i] in testval:
        ftest.write(data_list[i]+' '+data_list[i].split('/')[3]+'\n')
    else:
        ftrain.write(data_list[i]+' '+data_list[i].split('/')[3]+'\n')