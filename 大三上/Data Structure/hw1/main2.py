"""
@Author: wty-yy
@Date: 2023-01-11 17:51:35
@LastEditTime: 2023-01-12 10:48:34
@Description: 
@ 使用BST建立单词索引表，测试文本为article.txt，
@ BST中key值表示单词，value值表示单词在文本中出现的行号
"""
# coding:UTF-8

import my_bst

def wash(word):  # 将其他字符都去掉, 只剩下拉丁字母
    while len(word):
        if not word[-1].isalpha():
            word = word[:-1]
        elif not word[0].isalpha():
            word = word[1:]
        else:
            break
    return word

# with open('article_Harry.txt', 'r', encoding='utf-8') as file:
with open('article.txt', 'r', encoding='utf-8', errors='ignore') as file:
    bst = my_bst.BST(val0=[])
    cnt = 0  # 用于记录行数
    while True:
        # print(cnt)
        line = file.readline()
        if not line:
            break
        cnt += 1
        line = wash(line)  # 将每一行也洗一下, 把多余的空格去掉
        words = line.split(' ')  # 将一行中的单词分离出来
        for key in words:  # 逐个遍历
            key = wash(key)  # 将key中的其他标点去掉, 只剩下单词
            if not len(key):  # 洗没了
                continue
            bst.insert(key, cnt)

with open("my_result2.txt", 'w') as file:
    bst.printInorder(file)
