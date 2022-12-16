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


os.environ["CUDA_VISIBLE_DEVICES"]='1'  # 指定显卡编号

encoder_handle = r'model/bert_encoder/'  # 读取bert模型
preprocesser_handle = r'model/bert_preprocessor/'  # 读取预处理模型
ckp_load_handle = None  # 模型加载路径
ckp_save_handle = r'./checkpoints/bert_classifier'  # 模型保存路径
seq_length = 128  # 预处理文本最大长度

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
        data_y.append((1, class2idx[name]))
    if neg.shape[0] < 5000:
        add = neg.sample(5000 - neg.shape[0], replace=True)
        neg = pd.concat([neg, add], ignore_index=True)
    for i in range(neg.shape[0]):
        text = neg.iloc[i][2]
        text = tf.constant(text, tf.string)
        data_x.append(text)
        data_y.append((0, class2idx[name]))
print(f"输入特征: {len(data_x)}, 输出标签：{len(data_y)}")

# 模型预处理
ds = tf.data.Dataset.from_tensor_slices((data_x, data_y))

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
    # x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x1 = layers.Dense(1, name='emotion')(x)
    x2 = layers.Dense(10, name='classifier')(x)
    return keras.Model(text_input, [x1, x2])

classifier_model = build_classifier()

# 超参数配置
batch_size = 16
batch_N = 100000 / batch_size
epochs = 2
optimizer = optimizers.Adam(learning_rate=1e-5)  # Adam优化器，设定步长
binary_loss = losses.BinaryCrossentropy(from_logits=True)  # 二元交叉熵
multi_loss = losses.SparseCategoricalCrossentropy(from_logits=True)  # 多元交叉熵
dataset = ds.shuffle(100000).batch(batch_size).repeat(epochs)  # 随机打乱样本，设定batch大小
# 计数器
emotion_acc = keras.metrics.BinaryAccuracy('emotion_acc')  # 情感分类上的准确率
emotion_loss= keras.metrics.Mean('emotion_loss', dtype=tf.float32)  # 情感分类的平均损失
class_acc = keras.metrics.SparseCategoricalAccuracy('class_acc')  # 物品分类上的准确率
class_loss = keras.metrics.Mean('class_loss', dtype=tf.float32)  # 物品分类上的平均损失
metrics = [emotion_acc, emotion_loss, class_acc, class_loss]

if ckp_load_handle is not None:
    classifier_model.load_weights(ckp_load_handle)  # 从之前的继续进行训练

history = dict([(metric.name, []) for metric in metrics])
print("Start training!!!")
for step, (x, y) in tqdm(enumerate(dataset)):
    emotion_y = tf.reshape(y[:, 0], [-1, 1])  # 情感标签
    classes_y = tf.reshape(y[:, 1], [-1, 1])  # 分类标签
    with tf.GradientTape() as tape:
        emotion, classes = classifier_model(x, training=True)  # 模型预测
        loss1 = binary_loss(emotion_y, emotion)  # y=(batch, 2)
        loss2 = multi_loss(classes_y, classes)
        loss = tf.reduce_mean(loss1 + loss2)  # 将两个loss求和作为总损失
    grads = tape.gradient(loss, classifier_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, classifier_model.trainable_variables))

    emotion_acc.update_state(emotion_y, emotion)
    emotion_loss.update_state(loss1)
    class_acc.update_state(classes_y, classes)
    class_loss.update_state(loss2)

    if step % 100 == 0:
        s = f"step={step}/{batch_N * epochs}: "
        for metric in metrics:
            s += f"{metric.name}={metric.result().numpy():.3f} "
            history[metric.name].append(metric.result().numpy())
            metric.reset_states()
        print(s)
        figure = plt.figure()
        plt.plot(history['emotion_acc'], label='emotion_acc')
        plt.plot(history['class_acc'], label='class_acc')
        plt.legend()
        plt.title('Accuracy')
        figure.tight_layout()
        plt.savefig('acc.png', dpi=300)
        plt.close()

        figure = plt.figure()
        plt.plot(history['emotion_loss'], label='emotion_loss')
        plt.plot(history['class_loss'], label='class_loss')
        plt.legend()
        plt.title('Loss')
        figure.tight_layout()
        plt.savefig('loss.png', dpi=300)
        plt.close()

        classifier_model.save_weights(ckp_save_handle)


"""
两者同时进行训练

1个dense层：训练3125次，emotion_acc=0.952，emotion_loss=0.127
class_acc=0.919, class_loss=0.251

训练3125+6250次，emotion_acc=0.948，emotion_loss=0.134
class_acc=0.922, class_loss=0.237
---
2个dense层：训练6200次，emotion_acc=0.931，emotion_loss=0.194
class_acc=0.902，class_loss=0.310
---
预处理序列长度为256：batch_size=16，训练6250次，emotion_acc=0.933，emotion_loss=0.193
class_acc=0.898，class_loss=0.321

训练3个epochs，6250+12500：第后两个epoch降低learning_rate=1e-5，emotion_acc=0.966，emotion_loss=0.098
class_acc=0.944，class_loss=0.154  效果很好了
"""
