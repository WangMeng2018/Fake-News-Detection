import os,sys


def check_validation(source_path):

    f = open(source_path,"r")
    lines = f.readlines()[1:]
    f.close()

    label_set = set()
    text_list = list()
    length_set = set()
    i = 1
    for line in lines:
        tmps = line.strip().split('\t')
        if len(tmps) != 3:
            length_set.add(i)
        if len(tmps[1]) <= 5:
            text_list.append(tmps[1])
        label_set.add(tmps[2])
        i = i + 1
    
    print(len(length_set))
    print(length_set)

    print(len(text_list))
    print(text_list)

    print(len(label_set))
    print(label_set)

if __name__ == "__main__":
    check_validation("data/test.tsv")