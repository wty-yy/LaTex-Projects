# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: main.py.py
@time: 2022/11/30 17:26
"""
import collections


def solve(text):
    # 1.词库读取
    dic = set()
    with open('dict.txt_2.big', encoding='utf-8') as file:
        for s in file.readlines():
            word = s.split()[0]
            dic.add(word)
    print('词库大小', len(dic))

    counter = collections.Counter()
    for w in dic:
        counter[w] = len(w)
    print(counter.most_common()[:10])

    length = len(text)  # 获取文本长度
    print('原文本长度', length, '\n')

    def check(word):  # 判断是否为词语
        return word in dic

    # 2.正向最大匹配法
    pos_div= []
    pos_single = 0
    i = 0  # 左指针
    while i < length:
        for j in range(min(i+16, length), i, -1):  # 右指针
            if check(text[i:j]):
                pos_div.append(text[i:j])
                break
        if i + 1 == j:  # 记录单个字词个数
            pos_single += 1
        i = j
    print('正向最大匹配法长度：', len(pos_div))
    print('单个字个数：', pos_single)
    print(pos_div)

    # 3.逆向最大匹配法
    neg_div= []
    neg_single = 0
    j = length  # 右指针
    while j > 0:
        for i in range(max(j-16, 0), j):  # 左指针
            if check(text[i:j]):
                neg_div.append(text[i:j])
                break
        if i + 1 == j:  # 记录单个字词个数
            neg_single += 1
        j = i
    neg_div = neg_div[::-1]
    print('逆向最大匹配法长度：', len(neg_div))
    print('单个字个数：', neg_single)
    print(neg_div)
    print()

    # 4.双向最大匹配法
    best_div = None
    info = [len(pos_div)-len(neg_div), pos_single-neg_single]
    # 优先选取分词个数少的，再选取单个词少的
    if info[0] < 0 or (info[0] == 0 and info[1] < 0):
        best_div = pos_div
        print('选择正向')
    elif info[0] == info[1] == 0:  # 两个数目都相同
        best_div = pos_div
        print('两者相同')
    else:
        print('选择逆向')
        best_div = neg_div
    print(best_div)

if __name__ == '__main__':
    with open('text.txt', encoding='utf-8') as file:
        long_text = file.read()
    text = '我们在野生动物园玩'  # 设定目标文本
    # solve(text)
    solve(long_text)
