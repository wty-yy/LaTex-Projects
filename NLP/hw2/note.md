## 数据集

网址：http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes.html

包含两个新闻数据集：

1. 总数据集'20_newsgroups'：包含20种不同的新闻类别，总计共有19997篇文档，每种类别下应该平均有1000份新闻文档.
2. 子类文档'mini_newsgroups'：由第一个总数据集中每种类别的新闻中随机选择100份，总计2000份文档，用于验证算法的准确度.

## 数据预处理

### 文件读入处理

将20种文件类型进行编号，并查看内部的文档数目

```
                    Class  Id Files  Words  Most common words
              alt.atheism:  0  1000  10950  ['write', 'say', 'one', 'god', 'would']
            comp.graphics:  1  1000  13406  ['imag', 'file', 'use', 'program', 'write']
  comp.os.ms-windows.misc:  2  1000  48850  ['max', 'g', 'r', 'q', 'p']
 comp.sys.ibm.pc.hardware:  3  1000  10353  ['drive', 'use', 'get', 'card', 'scsi']
    comp.sys.mac.hardware:  4  1000   9354  ['use', 'mac', 'get', 'write', 'appl']
           comp.windows.x:  5  1000  20392  ['x', 'use', 'window', 'file', 'program']
             misc.forsale:  6  1000  10830  ['new', 'sale', 'offer', 'use', 'sell']
                rec.autos:  7  1000  10378  ['car', 'write', 'get', 'articl', 'would']
          rec.motorcycles:  8  1000  10207  ['write', 'bike', 'get', 'articl', 'dod']
       rec.sport.baseball:  9  1000   9164  ['game', 'year', 'write', 'good', 'get']
         rec.sport.hockey: 10  1000  11311  ['game', 'team', 'play', 'go', 'get']
                sci.crypt: 11  1000  13087  ['key', 'use', 'encrypt', 'would', 'write']
          sci.electronics: 12  1000  10480  ['use', 'one', 'would', 'write', 'get']
                  sci.med: 13  1000  15271  ['use', 'one', 'write', 'get', 'articl']
                sci.space: 14  1000  13867  ['space', 'would', 'write', 'orbit', 'one']
   soc.religion.christian: 15   997  12616  ['god', 'christian', 'one', 'would', 'say']
       talk.politics.guns: 16  1000  14626  ['gun', 'would', 'write', 'peopl', 'articl']
    talk.politics.mideast: 17  1000  15105  ['armenian', 'say', 'peopl', 'one', 'write']
       talk.politics.misc: 18  1000  13727  ['would', 'write', 'peopl', 'say', 'articl']
       talk.religion.misc: 19  1000  12390  ['write', 'say', 'one', 'god', 'would']
                              19997 146437  ['write', 'would', 'one', 'use', 'get']
```



### 分词操作

首先将20类的文档全部读入，将数据的主要成分提取出来，然后利用NLTK库的分词功能

1. 将文章转化为小写 `words.lower()`
2. 划分 `nltk.word_tokenize(words)`
3. 标点符号去除，用正则表达式判断单词中是否包含英文，若不包含则删去
4. 去除停用词，利用 `nltk.corpus.stopwords('english')` 获得停用词词库
5. 词干提取，使用 `nltk.stem.porter.PorterStemmer(word)` 词干提取方法
6. 词性还原，使用 `nltk.stem.WordNetLemmatizer(word)` 还原词性



## 分类模型

### K近邻

选择前1000个出现频率最高的单词作为词向量的基：

```
write, would, one, use, get, articl, say, know, like, think, make, peopl, good, go, time, x, see, also, could, work, u, take, right, new, want, system, even, way, year, thing, come, well, find, may, give, look, need, god, problem, much, mani, tri, first, two, file, mean, max, believ, call, run, question, point, q, anyon, post, seem, program, state, window, tell, differ, r, drive, read, realli, someth, plea, includ, g, sinc, thank, number, p, ca, back, univers, still, govern, reason, help, inform, day, start, person, game, gener, part, follow, might, support, c, law, sure, last, long, ask, case, fact, never, do, let, interest, set, christian, must, without, possibl, hear, group, comput, power, anoth, someon, car, avail, lot, n, b, name, show, put, keep, key, imag, line, great, exist, chang, live, send, actual, word, world, control, place, claim, high, list, happen, probabl, anyth, etc, data, howev, around, book, w, littl, opinion, v, everi, bite, card, kill, true, consid, least, cours, object, play, buy, best, child, softwar, space, idea, enough, base, talk, order, team, old, second, gun, end, provid, late, issu, version, el, nation, build, human, though, armenian, exampl, either, far, refer, public, k, note, david, noth, life, jesu, hard, sourc, respons, requir, ye, rather, wrong, real, answer, bad, caus, subject, understand, e, discus, report, origin, current, l, effect, john, f, valu, mail, quit, big, chip, man, allow, free, price, pay, other, h, care, standard, sever, le, kind, feel, moral, mayb, hope, suggest, machin, abl, evid, email, ever, alway, american, phone, next, turn, yet, result, engin, address, sell, disk, fire, hand, scienc, develop, sound, author, accept, whether, bill, small, close, type, applic, agre, forc, three, z, user, hold, lead, rememb, e-mail, internet, view, open, away, jew, j, import...
```

取K=5的分类结果

```
第1组类别，正确率: 0.55
第2组类别，正确率: 0.43
第3组类别，正确率: 0.55
第4组类别，正确率: 0.51
第5组类别，正确率: 0.38
第6组类别，正确率: 0.56
第7组类别，正确率: 0.44
第8组类别，正确率: 0.42
第9组类别，正确率: 0.46
第10组类别，正确率: 0.67
第11组类别，正确率: 0.57
第12组类别，正确率: 0.7
第13组类别，正确率: 0.62
第14组类别，正确率: 0.51
第15组类别，正确率: 0.57
第16组类别，正确率: 0.57
第17组类别，正确率: 0.5
第18组类别，正确率: 0.56
第19组类别，正确率: 0.72
第20组类别，正确率: 0.43
```



### SVM（支持向量机，线性分类器）

