import argparse
import os
import sys
import torch
import torch.nn.functional as F
import data_processor
from model import TextCNN


print("开始加载配置信息")
parser = argparse.ArgumentParser(description='TextCNN text classifier')
parser.add_argument('-lr', type=float, default=0.001, help='学习率')
parser.add_argument('-batch-size', type=int, default=32)  #128
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-filter-num', type=int, default=50, help='卷积核的个数') #100
parser.add_argument('-filter-sizes', type=str, default='3,4,5', help='不同卷积核大小')
parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-label-num', type=int, default=2, help='标签个数')
parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=100,help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')
args = parser.parse_args()
print("成功加载配置信息")

def predict(args):
    train_iter, dev_iter, test_iter = data_processor.load_data(args) # 将数据分为训练集和验证集
    print('加载测试数据完成')

    model = TextCNN(args)
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.load_state_dict(torch.load("/home/trec6/wangmeng/FND/model_dir/best_steps_3300.pt"))
    model.eval()

    for batch in test_iter:
        feature, text_id = batch.text, batch.id
        feature = feature.data.t()
        print(text_id, feature)
        if args.cuda:
            feature, text_id = feature.cuda(), text_id.cuda()
        logits = model(feature)
        print(logits)

predict(args)