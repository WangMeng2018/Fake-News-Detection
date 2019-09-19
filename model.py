import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self,args):
        super(TextCNN,self).__init__()
        self.args = args

        label_num = args.label_num  #类别个数
        filter_num = args.filter_num   #卷积核的个数
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]
        vocab_size = args.vocab_size
        embedding_dim = args.embedding_dim

        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim))for fsz in filter_sizes]
        )

        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_sizes)*filter_num, label_num)

    def forward(self,x):
        # input x:[batch_size, max_len]
        # output :[batch_size, max_len, embedding_dim]
        # max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)

        # view output:[batch_size, input_chanel=1, w=max_len, h=embedding_dim]
        x = x.view(x.size(0), 1, x.size(1), self.args.embedding_dim)

        # output:[batch_size, out_chanel, w, h=1]
        x = [F.relu(conv(x)) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)

        # dropout层
        x = self.dropout(x)

        # 全连接层
        logits = self.linear(x)
        return logits