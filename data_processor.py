import jieba
from torchtext import data
import re
from torchtext.vocab import Vectors


def tokenizer(text): # create a tokenizer function
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

def load_data(args):
    print('加载数据中...')
    stop_words = get_stop_words() # 加载停用词表
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    # id = data.Field(sequential=False)
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer, stop_words=stop_words)
    label = data.Field(sequential=False)

    text.tokenize = tokenizer
    train, val, test = data.TabularDataset.splits(
            path='data/',
            skip_header=True,
            train='train.tsv',
            validation='validation.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('id', None), ('text', text), ('label', label)],
    )

    if args.static:
        text.build_vocab(train, val, test, vectors=Vectors(name="data/eco_article.vector")) # 此处改为你自己的词向量
        args.embedding_dim = text.vocab.vectors.size()[-1]
        args.vectors = text.vocab.vectors

    else: text.build_vocab(train, val, test)
    
    # id.build_vocab(test)
    label.build_vocab(train, val, test)

    train_iter, val_iter, test_iter = data.Iterator.splits(
            (train, val, test),
            sort_key=lambda x: len(x.text),
            batch_sizes=(args.batch_size, len(val), len(test)), # 训练集设置batch_size,验证集整个集合用于测试
            device=-1
    )
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)
    print(args.label_num, label.vocab)
   
    return train_iter, val_iter, test_iter
