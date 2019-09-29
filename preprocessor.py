import csv
import pandas as pd


# 将原始CSV文件中空格、制表符、换行符等全部删除
def transform_text(source_path, train_path):
    csv_data = pd.read_csv(source_path)  # 读取训练数据
    print(csv_data.shape)

    nums = csv_data.shape[0]
    for i in range(nums):
        tmp_str = csv_data.loc[i,'text']
        tmp_str = "".join(tmp_str.strip().split(' '))
        csv_data.loc[i,'text'] = tmp_str

    csv_data.to_csv(train_path, index = False, header = True)    

#切分训练集和验证集
def split_text(source_path, train_path, valid_path, valid_num):
    r = open(source_path,'r')
    lines = r.readlines()
    r.close()

    train_lines = lines[valid_num:]
    f = open(train_path,'w')
    line = lines[0]
    line = "\t".join(line.strip().split(','))
    f.write(line + '\n')
    for line in train_lines:
        line = "\t".join(line.strip().split(','))
        f.write(line + '\n')
    f.close()        

    val_lines = lines[:valid_num]
    f = open(valid_path, 'w')
    for line in val_lines:
        line = "\t".join(line.strip().split(','))
        f.write(line + '\n')
    f.close()

def transform_test(source_path, target_path):
    r = open(source_path,'r')
    lines = r.readlines()[1:]
    r.close()

    f = open(target_path, 'w')
    f.write("id" + '\t' + "text" + '\t' + "label" + '\n')
    for line in lines:
        tmps = line.strip().split(',')
        tmp_str = tmps[1]
        tmp_str = "".join(tmp_str.strip().split(' \t\n'))
        tmps[1] = tmp_str
        line = "\t".join(tmps)
        f.write(line + '\t' + "0" + '\n')           # 测试集中的默认label是0，没有实际用处
    f.close()

if __name__ == "__main__":
    # transform_text("data/train.csv", "data/train_all.tsv")
    # split_text("data/train_all.tsv", "data/train.tsv", "data/validation.tsv", 5001)
    transform_test("data/test_stage1.csv", "data/test.tsv")