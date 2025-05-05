from gensim.models import Word2Vec

# 简单的语料
sentences = [
    ["国王", "是", "男人"],
    ["王后", "是", "女人"],
    ["王子", "是", "国王", "的", "儿子"],
    ["公主", "是", "王后", "的", "女儿"]
]

# 训练 Word2Vec 模型
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# 找出与“国王”最相近的词
print(model.wv.most_similar("国王"))
