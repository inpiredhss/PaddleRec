from __future__ import print_function
import random
import pickle

random.seed(1234)

print("read and process data")

with open('./raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    #print("cate_list",cate_list)
    user_count, item_count, cate_count, example_count = pickle.load(f)

#print("---reviews_df",reviews_df)
#print("---cate_list",cate_list)

train_set = []
test_set = []
j=0
for reviewerID, hist in reviews_df.groupby('reviewerID'):
 #   print("----reviewerID",reviewerID)
  #  print("----hist",hist)
    pos_list = hist['asin'].tolist()
   # print("----pos_list",pos_list)
   # print("----reviewerID",reviewerID)
    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    neg_list = [gen_neg() for i in range(len(pos_list))]
    #print("---neg_list",neg_list)
    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        if i != len(pos_list)//2:
            train_set.append((hist, pos_list[i], neg_list[i]))
        else:
            user_seq = pos_list[:len(pos_list)//2]
            item_labels = pos_list[len(pos_list)//2:]
            test_set.append((user_seq, item_labels))
    j=j+1
    if j>999:
        break
print("train_set----", train_set)
print("test_set----", test_set)
random.shuffle(train_set)
random.shuffle(test_set)

print("len(test_set)----",len(test_set))
print("user_count----",user_count)
#assert len(test_set) == user_count

def print_to_last_file(data, fout):
    for i in range(len(data)):
        fout.write(str(data[i]))
        if i != len(data) - 1:
            fout.write(' ')
        else:
            fout.write('\n')

def print_to_mid_file(data, fout):
    for i in range(len(data)):
        fout.write(str(data[i]))
        if i != len(data) - 1:
            fout.write(' ')
        else:
            fout.write(';')


print("make train data")
with open("v1_demo_train.txt", "w") as fout:
    for line in train_set:
        history = line[0]
        pos = line[1]
        neg = line[2]
        cate = [cate_list[x] for x in history]
        print_to_mid_file(history, fout)
        print_to_mid_file(cate, fout)
        fout.write(str(pos) + ";")
        fout.write(str(neg) + "\n")

print("make test data")
with open("v1_demo_test.txt", "w") as fout:
    for line in test_set:
        user_seq = line[0]
        label_items = line[1]
        user_cat = [cate_list[x] for x in user_seq]
        print_to_mid_file(user_seq, fout)
        print_to_mid_file(user_cat, fout)
        print_to_last_file(label_items, fout)
       # fout.write(str(target[0]) + ";")
       # fout.write("\n")

print("make config data")
with open('config_demo.txt', 'w') as f:
    f.write(str(user_count) + "\n")
    f.write(str(item_count) + "\n")
    f.write(str(cate_count) + "\n")
