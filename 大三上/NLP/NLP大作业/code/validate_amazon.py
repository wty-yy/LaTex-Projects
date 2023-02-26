#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential, losses, optimizers, metrics
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DIVICES"]='0'

encoder_handle = r'model/bert_encoder/'
preprocesser_handle = r'model/bert_preprocessor/'
# ckp_handle = r'./checkpoints/bert_classifier'
ckp_handle = r'./checkpoints2/bert_classifier_seq_256'
# ckp_handle = r'./checkpoints2/bert_classifier_epoch2'
# ckp_handle = r'./checkpoints/bert_classifier_seq_256_epoch2'
seq_length = 128

df = pd.read_csv(r'online_shopping_10_cats.csv')
class_names = list(df.cat.drop_duplicates())
class2idx = {}
idx2class = {}
for idx, name in enumerate(class_names):
    class2idx[name] = idx
    idx2class[idx] = name

# 自定义预处理模型
def bert_preprocessor(sentence_features, seq_length=128):
    text_inputs = [layers.Input(shape=(), dtype=tf.string, name=ft)
                   for ft in sentence_features]  # 处理输入的句子特征
    
    preprocessor = hub.load(preprocesser_handle)
    tokenize = hub.KerasLayer(preprocessor.tokenize, name='tokenizer')
    tokenized_inputs = [tokenize(segment) for segment in text_inputs]  # 将句子划分为字
    
    packer = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name='packer'
    )
    encoder_inputs = packer(tokenized_inputs)
    return keras.Model(text_inputs, encoder_inputs, name='preprocessor')
preprocessor = bert_preprocessor(['input1'], seq_length=seq_length)

def build_classifier():
    text_input = layers.Input(shape=(), dtype=tf.string, name='input')
    text_preprocessed = preprocessor(text_input)
    encoder = hub.KerasLayer(encoder_handle, trainable=True, name='BERT_encoder')
    x = encoder(text_preprocessed)['pooled_output']
    x = layers.Dropout(0.3)(x)
    x1 = layers.Dense(1, name='emotion')(x)
    x2 = layers.Dense(10, name='classifier')(x)
    return keras.Model(text_input, [x1, x2])
classifier_model = build_classifier()

classifier_model.load_weights(ckp_handle)

cat_names = ['书籍', '平板', '手机', '水果', '洗发水', '热水器', '蒙牛', '衣服', '计算机', '酒店']

# 构造验证集，3分及以下为负面评论，4分及以上为正面评论
info_df = pd.DataFrame(columns=['类别', '总数目', '正例', '负例'])
val_x, val_y = [], []
x = []
for name in cat_names:
    df = pd.read_csv(f"./validation/{name}.csv")
    if df.shape[0] > 15000:  # 每种类别上限10000项
        df = df.sample(15000)
    pos_num = df[df['rating']>3].shape[0]  # 正例数目
    neg_num = df[df['rating']<=2].shape[0]  # 负例数目
    tot_num = pos_num + neg_num
    print(f"类别'{name}', 总数目{tot_num}, 正例{pos_num}, 负例{neg_num}")
    info_df.loc[info_df.shape[0]] = [name, tot_num, pos_num, neg_num]
    def add(row):
        if row['rating'] == 3:  # 如果是3分认为是中等评论，不纳入数据集
            return
        emotion = 1 if row['rating'] > 3 else 0
        x.append(row['comment'])
        val_x.append(tf.constant(row['comment'], tf.string))
        val_y.append((emotion, class2idx[name]))
    df.apply(add, axis=1)
print(f"验证集大小{len(val_x)}")

val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_book = val_ds.take(10000)
val_pad = val_ds.skip(10000).take(10000)
emotion_acc = keras.metrics.BinaryAccuracy('emotion_acc')  # 情感分类上的准确率
class_acc = [0, 0]  # 物品分类上的准确率

def update_acc(classes_y, classes, acc, num=3):  # 结果在预测前num大中则正确
    for b in range(classes.shape[0]):
        acc[1] += 1
        pred = classes[b]
        arg = np.argsort(pred.numpy())[::-1]
        for i in range(num):
            if classes_y[b] == arg[i]:
                acc[0] += 1
                break

for (x, y) in tqdm(val_ds.batch(128)):
    emotion_y = tf.reshape(y[:, 0], [-1, 1])  # 情感标签
    classes_y = tf.reshape(y[:, 1], [-1, 1])  # 分类标签
    emotion, classes = classifier_model(x, training=False)
    emotion_acc.update_state(emotion_y, emotion)
    update_acc(classes_y, classes, class_acc, num=1)

print(f"情感分类准确率: {emotion_acc.result().numpy():.2%}")
print(f"商品分类准确率: {class_acc[0]/class_acc[1]:.2%}")
print(f"总计验证数目：{emotion_acc.count.numpy()}")
"""
设置为前排名前三的认为正确
bert_classifier_epoch1
情感分类准确率: 73.95%
商品分类准确率: 84.04%

bert_classifier_epoch2
情感分类准确率: 78.74%
商品分类准确率: 81.75%

bert_classifier_seq_256
情感分类准确率: 76.38%
商品分类准确率: 83.71%
"""
