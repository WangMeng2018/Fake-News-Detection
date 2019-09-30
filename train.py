import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import data_processor
from model import TextCNN

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

train_iter, dev_iter, test_iter, vocab = data_processor.load_data(args) # 将数据分为训练集和验证集
print('加载数据完成')


def train(args):
    model = TextCNN(args)
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epoch + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            # t_()函数表示将(max_len, batch_size)转置为(batch_size, max_len)
            feature = feature.data.t()
            target = target.data.sub(1) # target减去1
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                # torch.max(logits, 1)函数：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        # raise KeyboardInterrupt
                        return

'''
对验证集进行测试 
'''
def eval(data_iter, model, args):
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        target = target.data.sub(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()) == target).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def predict(args):
    model = TextCNN(args)
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.load_state_dict(torch.load("./model_dir/best_steps.pt"))
    model.eval()

    for batch in test_iter:
        id, feature = batch.id, batch.text
        feature = feature.data.t()
        test_num = id.data.size()[0]
        id_list = [vocab.itos[id.data[i]] for i in range(test_num)]
        if args.cuda:
            id, feature = id.cuda(), feature.cuda()
        logits = model(feature)
        max_value, max_index = torch.max(logits, dim=1)
        result = max_index.numpy().tolist()
        # np.savetxt("result.txt", result, fmt='%d', delimiter=",")

        f = open("submit.csv","w")
        f.write("id" + ',' + "label" + '\n')
        for i in range(test_num):
            f.write(id_list[i] + ',' + str(result[i]) + '\n')
        f.close()

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    # save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    save_path = '{}_steps.pt'.format(save_prefix)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train(args)
    predict(args)
