from gensim.models import LdaModel
from gensim import corpora
import jieba
import re
import pandas as pd
from pprint import pprint

raw_data = pd.read_csv('./data/udn-news-topics.csv')
raw_corpus = raw_data.iloc[:149]['text'].tolist()
raw_test = raw_data.iloc[149:]['text'].tolist()
stop_words = [line.strip() for line in open('./cn_stopwords.txt', 'r', encoding='utf-8').readlines()]

corpus = []
for sentence in raw_corpus:
    corpus.append(
        [item for item in jieba.cut(sentence) if item not in stop_words and re.match(r"[\u4e00-\u9fa5]+", item)]
    )  # 去掉停止词，只保留中文

test_corpus = []
for sentence in raw_test:
    test_corpus.append(
        [item for item in jieba.cut(sentence) if item not in stop_words and re.match(r"[\u4e00-\u9fa5]+", item)]
    )  # 去掉停止词，只保留中文

dictionary = corpora.Dictionary(corpus)
dictionary.filter_extremes(no_below=1, no_above=0.3, keep_n=None)
dictionary.compactify()  # 去掉因删除词汇而出现的空白
dictionary.save("./data/news_dict.dict")  # 保存生成的词典

corpus_bow = [dictionary.doc2bow(s) for s in corpus]
test_bow = [dictionary.doc2bow(s) for s in test_corpus]

# Set training parameters.
num_topics = 16
iterations = 400

model = LdaModel(corpus=corpus_bow, num_topics=num_topics, id2word=dictionary, iterations=iterations)
model.save('./data/news.model')  # 将模型保存到硬盘
pprint(model.print_topics(num_topics=num_topics))

predictions = []
probabilities = []
for doc in corpus_bow:
    topics = model.get_document_topics(doc)
    topics.sort(key=lambda x: x[1], reverse=True)
    predictions.append(topics[0][0])
    probabilities.append(topics[0][1])
for doc in test_bow:
    topics = model.get_document_topics(doc)
    topics.sort(key=lambda x: x[1], reverse=True)
    predictions.append(topics[0][0])
    probabilities.append(topics[0][1])

raw_data['topic'] = predictions
raw_data['probability'] = probabilities

raw_data.to_csv('./udn-news-topics-lda.csv', index=False, encoding='utf-8-sig')

