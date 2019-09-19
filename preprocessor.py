def transform_text(source_path, target_path, valid_num):
    r = open(source_path,'r')
    lines = r.readlines()
    r.close()

    train_lines = lines[valid_num:]
    f = open(source_path,'w')
    line = lines[0]
    line = "\t".join(line.strip().split(','))
    f.write(line + '\n')
    for line in train_lines:
        line = "\t".join(line.strip().split(','))
        f.write(line + '\n')
    f.close()        

    val_lines = lines[:valid_num]
    f = open(target_path, 'w')
    for line in val_lines:
        line = "\t".join(line.strip().split(','))
        f.write(line + '\n')
    f.close()

if __name__ == "__main__":
    transform_text("data/train.tsv", "data/validation.tsv", 5001)