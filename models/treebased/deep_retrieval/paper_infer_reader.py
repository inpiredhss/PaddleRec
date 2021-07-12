from __future__ import print_function
import numpy as np
import io
import random
import paddle
from paddle.io import IterableDataset

class RecDataset(IterableDataset):
    def __init__(self, file_list, config, graph_index, mode = "train"):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.init()
        self.mode = mode
        self.graph_index = graph_index
        self.batch_size = config.get("runner.train_batch_size")
        self.group_size = (self.batch_size) * 2


    def init(self):
        self.res = []
        self.max_len = 0
        for file in self.file_list:
            with open(file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    user_seq = line[0].split()
                    self.max_len = max(self.max_len, len(user_seq))
        fo = open("tmp.txt", "w")
        fo.write(str(self.max_len))
        fo.close()
        pass

    def __iter__(self):
        file_dir = self.file_list
        res0 = []
        for infer_file in file_dir:
            with open(infer_file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    if len(line)<3:
                        continue
                    user_seq = line[0].split()
                    # print("-------user_seq", user_seq)
                    user_cat = line[1].split()
                    # print("-------user_cat", user_cat)
                    label_items = line[2].split()
                    # print("-------label_items", label_items)
                    res0.append([user_seq, user_cat,label_items])
                    # print("-------res0", res0)

        data_set = res0
        random.shuffle(data_set)

        reader, batch_size, group_size = data_set, self.batch_size, self.group_size
        bg = []
        for line in reader:
            bg.append(line)
            if len(bg) == group_size:  # #
                sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
                bg = []
                for i in range(0, group_size, batch_size):
                    # print("------first-------")
                    b = sortb[i:i + batch_size]
                    # print("-----b",b)
                    max_len = max(len(x[0]) for x in b) + 1
                    user_seq = [x[0] for x in b]
                    # print("--user_seq---", user_seq)
                    user_seq_mask = np.array(
                        [x + ['0'] * (max_len - len(x)) for x in user_seq]).astype("int64").reshape([-1, max_len])
                    # print("---user_seq_mask---", user_seq_mask)
                    user_cat = [x[1] for x in b]
                    user_cat_mask = np.array(
                        [x + ['0'] * (max_len - len(x)) for x in user_cat]).astype("int64").reshape([-1, max_len])
                    label_items = [x[2] for x in b]
                    # print("----label_items", label_items)

                    label_items_mask = np.array(
                        [x + ['0'] * (max_len - len(x)) for x in label_items]).astype("int64").reshape([-1, max_len])
                    # print("----label_items_mask",label_items_mask)

                    for i in range(len(b)):
                        res = []
                        # print("------second-------")
                        res.append(np.array(user_seq_mask[i]).astype('int64'))
                        res.append(np.array(user_cat_mask[i]).astype('int64'))
                        res.append(np.array(label_items_mask[i]).astype('int64')) 
                        # print("-------res", res)   
                        yield res

        len_bg = len(bg)
        if len_bg != 0:
            sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
            bg = []
            remain = len_bg % batch_size
            for i in range(0, len_bg - remain, batch_size):
                b = sortb[i:i + batch_size]
                max_len = max(len(x[0]) for x in b) + 1

                label_items = [x[2] for x in b]
                # print("----label_items", label_items)

                user_seq = [x[0] for x in b]
                # print("--user_seq---", user_seq)
                user_seq_mask = np.array(
                    [x + [0] * (max_len - len(x)) for x in user_seq]).astype("int64").reshape([-1, max_len])
                # print("---user_seq_mask---", user_seq_mask)
                user_cat = [x[1] for x in b]
                user_cat_mask = np.array(
                    [x + ['0'] * (max_len - len(x)) for x in user_cat]).astype("int64").reshape([-1, max_len])
                label_items = [x[2] for x in b]
                # print("----label_items", label_items)
                   
                label_items_mask = np.array(
                    [x + ["0"] * (max_len - len(x)) for x in label_items]).astype("int64").reshape([-1, max_len])
                # print("----label_items_mask",label_items_mask)


                for i in range(len(b)):
                    res = []
                    res.append(np.array(user_seq_mask[i]).astype('int64'))
                    res.append(np.array(user_cat_mask[i]).astype('int64'))
                    res.append(np.array(label_items_mask[i]).astype('int64'))   
                    # print("-------res", res) 
                    yield res
