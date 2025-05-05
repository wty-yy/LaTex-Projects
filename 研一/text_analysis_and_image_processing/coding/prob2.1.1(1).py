import jieba
from collections import Counter

poem = """床前明月光，疑是地上霜。
举头望明月，低头思故乡。"""

# 使用精确模式进行分词
words = jieba.lcut(poem, cut_all=False)

# 统计词频
counter = Counter(words)
print("分词结果：", words)
print("高频词：", counter.most_common(3))
