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

encoder_handle = r'model/bert_zh_L-12_H-768_A-12_4/'
preprocesser_handle = r'model/bert_zh_preprocess_3/'

# 数据预处理
df = pd.read_csv(r'../dataset/online_shopping_10_cats.csv')
class_names = list(df.cat.drop_duplicates())
info_df = pd.DataFrame(columns=['类别', '总数目', '正例', '负例'])
class2idx = {}
for idx, name in enumerate(class_names):
    tmp = df[df.cat==name]
    info_df.loc[info_df.shape[0]] = [name, tmp.shape[0], tmp[tmp.label==1].shape[0], tmp[tmp.label==0].shape[0]]
    class2idx[name] = idx
info_df

# 平衡每种类别的商品数目到10000正负例均为5000，不足5000则重复性随机选取，补齐到5000
data_x, data_y = [], []
for name in tqdm(class_names):
    tmp = df[df.cat==name]
    pos, neg = tmp[tmp.label==1], tmp[tmp.label==0]
    if pos.shape[0] < 5000:
        add = pos.sample(5000 - pos.shape[0], replace=True)
        pos = pd.concat([pos, add], ignore_index=True)
    for i in range(pos.shape[0]):
        text = pos.iloc[i][2]
        try:
            text = tf.constant(text, tf.string)
        except:
            print(text)
        data_x.append(text)
        data_y.append((class2idx[name], 1))
    if neg.shape[0] < 5000:
        add = neg.sample(5000 - neg.shape[0], replace=True)
        neg = pd.concat([neg, add], ignore_index=True)
    for i in range(neg.shape[0]):
        text = neg.iloc[i][2]
        text = tf.constant(text, tf.string)
        data_x.append(text)
        data_y.append((class2idx[name], 0))

# ## 模型预处理
ds = tf.data.Dataset.from_tensor_slices((data_x, data_y))

# 建立预处理模型
# 自定义预处理模型
def bert_preprocessor(sentence_features, seq_length=256):
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

keras.utils.plot_model(preprocessor, show_shapes=True, show_dtype=True, to_file='Preprocessor.png')

bert_model = hub.KerasLayer(encoder_handle)

def build_classifier():
    class Classifier(keras.Model):
        def __init__(self):
            super().__init__(name='prediction')
            self.encoder = bert_model = hub.KerasLayer(encoder_handle, trainable=False)
            # self.dense = layers.Dense(768, activation='relu')
            self.dropout = layers.Dropout(0.3)
            self.emotion = layers.Dense(1, activation='softmax')  # 情感分类
            self.classifier = layers.Dense(10, activation='softmax')  # 文本分类
            
        def call(self, text):  # 经过预处理后的文本
            output = self.encoder(text)
            pooled_output = output['pooled_output']
            x = self.dropout(pooled_output)
            x1 = self.emotion(x)
            x2 = self.classifier(x)
            return (x1, x2)
    
    model = Classifier()
    return model

model = build_classifier()

# 超参数配置
batch_size = 32
batch_N = 100000 / 32
epochs = 10
optimizer = optimizers.Adam(learning_rate=1e-4)  # Adam优化器，设定步长
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)  # 使用交叉熵损失函数
dataset = ds.shuffle(100000).batch(batch_size).repeat(epochs)  # 随机打乱样本，设定batch大小
# 计数器
emotion_acc = keras.metrics.SparseCategoricalAccuracy('emotion_acc')  # 情感分类上的准确率
emotion_loss= keras.metrics.Mean('emotion_loss', dtype=tf.float32)  # 情感分类的平均损失
class_acc = keras.metrics.SparseCategoricalAccuracy('class_acc')  # 物品分类上的准确率
class_loss = keras.metrics.Mean('class_loss', dtype=tf.float32)  # 物品分类上的平均损失
metrics = [emotion_acc, emotion_loss, class_acc, class_loss]

history = dict([(metric.name, []) for metric in metrics])
for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:
        x_preprocessed = preprocessor(x)  # 特征编码
        out = model(x_preprocessed)  # 模型预测
        emotion, classes = out[:, 0], out[:, 1:]  # 将第一维作为情感分类，后面10维作为文本分类
        loss1 = loss_fn(y[0], emotion)
        loss2 = loss_fn(y[1], classes)
        loss = tf.reduce_sum(loss1, loss2)  # 将两个loss求和作为总损失
    grads = tape.gradient(loss, model.trainable_variables)  # 求梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 更新网络参数

    emotion_acc.update_state(y[0], emotion)
    emotion_loss.update_state(loss1)
    class_acc.update_state(y[1], classes)
    class_loss.update_state(loss2)

    if step % 100 == 0:
        s = f"step={step}/{batch_N * epochs}: "
        for metric in metrics:
            s += f"{metric.name}={metric.result} "
            history[metric.name].append(metric.result)
            metric.reset_states()
