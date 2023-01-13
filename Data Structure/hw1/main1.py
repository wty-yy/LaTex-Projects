"""
@Author: wty-yy
@Date: 2023-01-11 17:37:31
@LastEditTime: 2023-01-12 09:33:54
@Description: 
@ 从测试命令文本BST_textcases.txt中读取命令，
@ 打印输出到文件my_result中，最后与答案文件BST_result.txt进行比对
"""
# coding:UTF-8
import my_bst

with open('BST_testcases.txt', 'r', encoding='utf-8') as file,\
     open('my_result1.txt', 'w', encoding='utf-8') as outfile:
    bst = my_bst.BST()
    while True:
        line = file.readline()
        if not line:
            break
        opt = line[0]
        if opt == '#':
            bst.showStructure(outfile)
            continue
        key = line.split(' ')[1]  # 提取key值
        if opt == '+':
            value = line.split('\"')[1]  # 提取value值
            bst.insert(key, value)
        elif opt == '-':
            value = bst.remove(key)
            if value is not None:
                outfile.write(f'remove success ---{key} {value}\n')
            else:
                outfile.write(f'remove unsuccess ---{key}\n')
        elif opt == '?':
            value = bst.search(key)
            if value is not None:
                outfile.write(f'search success ---{key} {value}\n')
            else:
                outfile.write(f'search unsuccess ---{key}\n')
        elif opt == '=':
            value = line.split('\"')[1]  # 提取value值
            flag = bst.update(key, value)
            if flag:
                outfile.write(f'update success ---{key} {value}\n')
            else:
                outfile.write(f'update unsuccess ---{key}\n')
