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

os.environ["CUDA_VISIBLE_DEVICES"]='2'  # 指定显卡编号

encoder_handle = r'model/bert_encoder/'
preprocesser_handle = r'model/bert_preprocessor/'
ckp_handle = r'./checkpoints/bert_classifier_epoch2'
# ckp_handle = r'./checkpoints/bert_classifier_two_layers'

# 数据预处理
df = pd.read_csv(r'online_shopping_10_cats.csv')
df = df[df.review.isna() == False]  # 去掉Nan元素
class_names = list(df.cat.drop_duplicates())
info_df = pd.DataFrame(columns=['类别', '总数目', '正例', '负例'])
class2idx = {}
for idx, name in enumerate(class_names):
    tmp = df[df.cat==name]
    info_df.loc[info_df.shape[0]] = [name, tmp.shape[0], tmp[tmp.label==1].shape[0], tmp[tmp.label==0].shape[0]]
    class2idx[name] = idx

# 验证集无需进行补全，直接选取全部的数据进行验证
data_x, data_y = [], []
for name in tqdm(class_names):
    tmp = df[df.cat==name]
    pos, neg = tmp[tmp.label==1], tmp[tmp.label==0]
    for i in range(pos.shape[0]):
        text = pos.iloc[i][2]
        data_x.append(text)
        data_y.append((1, class2idx[name]))
    for i in range(neg.shape[0]):
        text = neg.iloc[i][2]
        text = tf.constant(text, tf.string)
        data_x.append(text)
        data_y.append((0, class2idx[name]))
print(f"输入特征: {len(data_x)}, 输出标签：{len(data_y)}")

# 模型预处理
ds = tf.data.Dataset.from_tensor_slices((data_x, data_y))

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
preprocessor = bert_preprocessor(['input1'])

def build_classifier():
    text_input = layers.Input(shape=(), dtype=tf.string, name='input')
    text_preprocessed = preprocessor(text_input)
    encoder = hub.KerasLayer(encoder_handle, trainable=True, name='BERT_encoder')
    x = encoder(text_preprocessed)['pooled_output']
    # x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x1 = layers.Dense(1, name='emotion')(x)
    x2 = layers.Dense(10, name='classifier')(x)
    return keras.Model(text_input, [x1, x2])

classifier_model = build_classifier()

# 计数器
emotion_acc = keras.metrics.BinaryAccuracy('emotion_acc')  # 情感分类上的准确率
class_acc = keras.metrics.SparseCategoricalAccuracy('class_acc')  # 物品分类上的准确率

classifier_model.load_weights(ckp_handle)  # 从之前的继续进行训练

val_ds = ds.shuffle(100000).batch(128)
emotion_acc.reset_states()
class_acc.reset_states()
for (x, y) in tqdm(val_ds):
    emotion_y = tf.reshape(y[:, 0], [-1, 1])  # 情感标签
    classes_y = tf.reshape(y[:, 1], [-1, 1])  # 分类标签
    emotion, classes = classifier_model(x, training=False)
    emotion_acc.update_state(emotion_y, emotion)
    class_acc.update_state(classes_y, classes)
print(f"情感分类准确率: {emotion_acc.result().numpy():.2%}")
print(f"商品分类准确率: {class_acc.result().numpy():.2%}")
print(f"总计验证数目：{emotion_acc.count.numpy()}")

"""
bert_classifier:
情感分类准确率：94.84%
商品分类准确率：90.34%

bert_classifier_epoch2:
情感分类准确率：95.90%
商品分类准确率：92.04%

bert_classifier_two_layers:
情感分类准确率：90.18%
商品分类准确率：87.48%
"""
