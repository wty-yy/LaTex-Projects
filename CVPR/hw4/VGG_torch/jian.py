root_path = 'data/256_ObjectCategories/'
config_path = 'train.txt'
files = open(config_path, 'r').read().split('\n')
img_path = []
label = []
for file in files:
    file = file.split('/')
    #print(file[-1])
    file_path = root_path + file[-2] + '/' + file[-1].split(' ')[0]
    class_name = int(file[-1].split(' ')[1].split('.')[0])
    print(file_path)
    img_path.append(file_path)
    label.append(class_name)