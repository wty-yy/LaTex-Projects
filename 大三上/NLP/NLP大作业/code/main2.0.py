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

config = {
    "font.family": 'serif', # 衬线字体
    "figure.figsize": (8, 6),  # 图像大小
    "font.size": 16, # 字号大小
    "mathtext.fontset": 'stix', # 渲染数学公式字体
    'axes.unicode_minus': False # 显示负号
}
plt.rcParams.update(config)

# 2.0在原有基础上加入数据集的划分，按照3:1划分为训练集与验证集，加入test函数用于对验证集进行验证

os.environ["CUDA_VISIBLE_DEVICES"]='0'  # 指定显卡编号

val_split = 25  # 验证集的占比
seed = 109  # 随机种子
encoder_handle = r'model/bert_encoder/'  # 读取bert模型
preprocesser_handle = r'model/bert_preprocessor/'  # 读取预处理模型
ckp_load_handle = None  # 模型加载路径
ckp_save_handle = r'./checkpoints2/bert_classifier_layout1_epochs2'  # 模型保存路径
seq_length = 128  # 预处理文本最大长度
batch_size = 32  # batch大小
learning_rate = [1e-4, 1e-5]  # 动态调整步长，每个epoch调整一次
epochs = 2  # 重复训练epochs个训练集
train_flag = True  # 是否训练（只进行验证）

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

# 首先将每种类别商品的数据按照3:1划分为训练集和验证集，然后对训练集进行补全到7500个数据
train_ds, val_ds = None, None
for name in class_names:
    tmp = df[df.cat==name]
    pos_num, neg_num = tmp[tmp.label==1].shape[0], tmp[tmp.label==0].shape[0]
    pos_ds = tf.data.Dataset.from_tensor_slices((tmp[tmp.label==1]['review'], [(1, class2idx[name]) for _ in range(pos_num)]))
    neg_ds = tf.data.Dataset.from_tensor_slices((tmp[tmp.label==0]['review'], [(0, class2idx[name]) for _ in range(neg_num)]))
    
    pos_num = pos_num * val_split // 100
    neg_num = neg_num * val_split // 100
    pos_val = pos_ds.take(pos_num)
    pos_train = pos_ds.skip(pos_num)
    neg_val = neg_ds.take(neg_num)
    neg_train = neg_ds.skip(neg_num)
    
    train_marge = pos_train.concatenate(neg_train).shuffle(10000, seed=seed).repeat(-1)  # 合并正负数据，再补齐到7500个数据
    train_ds = train_marge.take(7500) if train_ds is None else train_ds.concatenate(train_marge.take(7500))
    val_ds = pos_val.concatenate(neg_val) if val_ds is None else val_ds.concatenate(pos_val).concatenate(neg_val)
    
    print(f"加入'{name}'后：训练集大小{train_ds.cardinality()}，验证集大小{val_ds.cardinality()}")

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

# 模型搭建
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
batch_N = train_ds.cardinality() // batch_size
optimizer = optimizers.Adam(learning_rate=learning_rate[0])  # Adam优化器，设定步长
binary_loss = losses.BinaryCrossentropy(from_logits=True)  # 二元交叉熵
multi_loss = losses.SparseCategoricalCrossentropy(from_logits=True)  # 多元交叉熵
train_ds = train_ds.shuffle(100000).batch(batch_size).repeat(epochs)  # 随机打乱样本，设定batch大小
val_ds = val_ds.shuffle(10000).batch(batch_size)
# 计数器
emotion_acc = keras.metrics.BinaryAccuracy('emotion_acc')  # 情感分类上的准确率
emotion_loss= keras.metrics.Mean('emotion_loss', dtype=tf.float32)  # 情感分类的平均损失
class_acc = keras.metrics.SparseCategoricalAccuracy('class_acc')  # 物品分类上的准确率
class_loss = keras.metrics.Mean('class_loss', dtype=tf.float32)  # 物品分类上的平均损失
metrics = [emotion_acc, emotion_loss, class_acc, class_loss]

if ckp_load_handle is not None:
    classifier_model.load_weights(ckp_load_handle)  # 从之前的继续进行训练

history = dict()
def draw_figure():  # 绘制训练过程结果
    figure = plt.figure()
    plt.plot(history['emotion_acc'], label='emotion_acc')
    plt.plot(history['class_acc'], label='class_acc')
    plt.plot(history['val_emotion_acc'], label='val_emotion_acc', ls='--')
    plt.plot(history['val_class_acc'], label='val_class_acc', ls='--')
    plt.legend()
    plt.title('Accuracy')
    figure.tight_layout()
    plt.savefig('acc.png', dpi=300)
    plt.close()

    figure = plt.figure()
    plt.plot(history['emotion_loss'], label='emotion_loss')
    plt.plot(history['class_loss'], label='class_loss')
    plt.plot(history['val_emotion_loss'], label='val_emotion_loss', ls='--')
    plt.plot(history['val_class_loss'], label='val_class_loss', ls='--')
    plt.legend()
    plt.title('Loss')
    figure.tight_layout()
    plt.savefig('loss.png', dpi=300)
    plt.close()

def test(val_ds):  # 模型测试
    for (x, y) in val_ds:
        emotion_y = tf.reshape(y[:, 0], [-1, 1])  # 情感标签
        classes_y = tf.reshape(y[:, 1], [-1, 1])  # 分类标签
        emotion, classes = classifier_model(x, training=False)
        loss1 = binary_loss(emotion_y, emotion)
        loss2 = multi_loss(classes_y, classes)

        emotion_acc.update_state(emotion_y, emotion)
        emotion_loss.update_state(loss1)
        class_acc.update_state(classes_y, classes)
        class_loss.update_state(loss2)

def train():  # 模型训练
    global optimizer  # 为了能修改全局变量
    for step, (x, y) in tqdm(enumerate(train_ds)):
        emotion_y = tf.reshape(y[:, 0], [-1, 1])  # 情感标签
        classes_y = tf.reshape(y[:, 1], [-1, 1])  # 分类标签
        with tf.GradientTape() as tape:
            emotion, classes = classifier_model(x, training=True)  # 模型预测
            loss1 = binary_loss(emotion_y, emotion)  # y=(batch, 2)
            loss2 = multi_loss(classes_y, classes)
            loss = tf.reduce_mean(loss1 + loss2)  # 将两个loss求和作为总损失
        grads = tape.gradient(loss, classifier_model.trainable_variables)  # 求梯度
        optimizer.apply_gradients(zip(grads, classifier_model.trainable_variables))  # 更新网络参数

        emotion_acc.update_state(emotion_y, emotion)
        emotion_loss.update_state(loss1)
        class_acc.update_state(classes_y, classes)
        class_loss.update_state(loss2)

        if step % 100 == 0:
            s = f"step={step}/{batch_N * epochs}: "
            for metric in metrics:
                s += f"{metric.name}={metric.result().numpy():.3f} "
                if metric.name not in history.keys():
                    history[metric.name] = []
                history[metric.name].append(metric.result().numpy())
                metric.reset_states()
            print(s)
            test(val_ds.take(100))  # 求验证集上的准确率，选取100*batch_size个的数据进行验证
            s = ""
            for metric in metrics:
                s += f"{'val_'+metric.name}={metric.result().numpy():.3f} "
                if 'val_' + metric.name not in history.keys():
                    history['val_'+metric.name] = []
                history['val_'+metric.name].append(metric.result().numpy())
                metric.reset_states()
            print(s)
            draw_figure()  # 绘制结果图
            classifier_model.save_weights(ckp_save_handle)  # 保存模型权重
            print(f"Save in '{ckp_save_handle}'")

        if step % batch_N == 0:  # 调整步长
            if len(learning_rate)-1 < step // batch_N:
                continue
            optimizer = optimizers.Adam(learning_rate=learning_rate[step//batch_N])
            print("调整步长为", learning_rate[step//batch_N])

if train_flag:
    print(f"Start training!!!")
    train()

print("整个验证集上测试")
for metric in metrics:
    metric.reset_states()
test(val_ds)
print(f"情感分类准确率: {emotion_acc.result().numpy():.2%}")
print(f"商品分类准确率: {class_acc.result().numpy():.2%}")
print(f"总计验证数目：{emotion_acc.count.numpy()}")

"""
bert_classifier：dropout=0.3 epochs=1
step=2300/2343: emotion_acc=0.944 emotion_loss=0.165 class_acc=0.919 class_loss=0.248
val_emotion_acc=0.940 val_emotion_loss=0.168 val_class_acc=0.877 val_class_loss=0.354
情感分类准确率: 93.67%
商品分类准确率: 88.62%

bert_classifier: dropout=0.3 epochs=2
step=4600/4686: emotion_acc=0.978 emotion_loss=0.070 class_acc=0.952 class_loss=0.130
val_emotion_acc=0.956 val_emotion_loss=0.142 val_class_acc=0.893 val_class_loss=0.318
4688it [45:32,  1.72it/s]
情感分类准确率: 94.57%
商品分类准确率: 90.63%

bert_classifier: dropout=0.3 epochs=3
step=2300/2343: emotion_acc=0.982 emotion_loss=0.060 class_acc=0.965 class_loss=0.100
val_emotion_acc=0.949 val_emotion_loss=0.180 val_class_acc=0.892 val_class_loss=0.346
Save in './checkpoints2/bert_classifier_epoch3'
2344it [24:03,  1.62it/s]
情感分类准确率: 94.70%
商品分类准确率: 90.92%

bert_classifier: dropout=0.3 epochs=4
step=2300/2343: emotion_acc=0.987 emotion_loss=0.046 class_acc=0.969 class_loss=0.082
val_emotion_acc=0.945 val_emotion_loss=0.203 val_class_acc=0.894 val_class_loss=0.410
Save in './checkpoints2/bert_classifier_epoch4'
2344it [24:24,  1.60it/s]
整个验证集上测试
情感分类准确率: 94.42%
商品分类准确率: 90.97%

bert_classifier: dropout=0.1, epochs=1
step=2300/2343: emotion_acc=0.944 emotion_loss=0.149 class_acc=0.915 class_loss=0.249
val_emotion_acc=0.937 val_emotion_loss=0.209 val_class_acc=0.869 val_class_loss=0.369
情感分类准确率: 93.09%
商品分类准确率: 88.56%

bert_classifier: dropout=0.1, epochs=2
step=4600/4686: emotion_acc=0.966 emotion_loss=0.095 class_acc=0.948 class_loss=0.146
val_emotion_acc=0.943 val_emotion_loss=0.185 val_class_acc=0.895 val_class_loss=0.305
4688it [45:30,  1.72it/s]
情感分类准确率: 93.99%
商品分类准确率: 90.59%

bert_classifier: dropout=0.3, epochs=2, seq_length=256, batch_size=16
step=9300/9374: emotion_acc=0.957 emotion_loss=0.114 class_acc=0.940 class_loss=0.164
val_emotion_acc=0.936 val_emotion_loss=0.191 val_class_acc=0.871 val_class_loss=0.390
情感分类准确率: 93.26%
商品分类准确率: 89.45%
"""
