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
        self.use_multi_task_learning = config.get("hyper_parameters.use_multi_task_learning")
        self.item_count = config.get("hyper_parameters.item_count")
        self.mode = mode
        self.graph_index = graph_index
        self.batch_size = config.get("runner.train_batch_size")
        self.group_size = (self.batch_size) * 5

    def init(self):
        self.res = []
        self.max_len = 0
        self.neg_candidate_item = []
        self.neg_candidate_cat = []
        self.max_neg_item = 1000
        self.max_neg_cat = 1000

        for file in self.file_list:
            with open(file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    hist = line[0].split()
                    self.max_len = max(self.max_len, len(hist))
        fo = open("tmp.txt", "w")
        fo.write(str(self.max_len))
        fo.close()

    def __iter__(self):
        file_dir = self.file_list
        res0 = []
        for train_file in file_dir:
            with open(train_file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    if len(line)<5:
                        continue
                    hist = line[0].split()
                    cate = line[1].split()
                    res0.append([hist, cate, line[2], line[3], float(line[4])])
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
                    b = sortb[i:i + batch_size]
                    max_len = max(len(x[0]) for x in b)

                    itemInput = [x[0] for x in b]
                    itemRes0 = np.array(
                        [x + [0] * (max_len - len(x)) for x in itemInput])
                    item = itemRes0.astype("int64").reshape([-1, max_len])
                    catInput = [x[1] for x in b]
                    catRes0 = np.array(
                        [x + [0] * (max_len - len(x)) for x in catInput])
                    cat = catRes0.astype("int64").reshape([-1, max_len])

                    len_array = [len(x[0]) for x in b]
                    mask = np.array(
                        [[0] * x + [-1e9] * (max_len - x)
                         for x in len_array]).reshape([-1, max_len, 1])
                    target_item_seq = np.array(
                        [[x[2]] * max_len for x in b]).astype("int64").reshape(
                            [-1, max_len])
                    target_cat_seq = np.array(
                        [[x[3]] * max_len for x in b]).astype("int64").reshape(
                            [-1, max_len])
                    
                    neg_item = [None] * len(item)
                    neg_cat = [None] * len(cat)

                    for i in range(len(b)):
                        neg_item[i] = []
                        neg_cat[i] = []
                        if len(self.neg_candidate_item) < self.max_neg_item:
                            self.neg_candidate_item.extend(b[i][0])
                            if len(self.neg_candidate_item) > self.max_neg_item:
                                self.neg_candidate_item = self.neg_candidate_item[
                                    0:self.max_neg_item]
                        else:
                            len_seq = len(b[i][0])
                            if self.max_neg_item < len_seq:
                                len_seq = len_seq % self.max_neg_item
                            # if self.max_neg_item > len_seq:
                            start_idx = random.randint(0, self.max_neg_item - len_seq - 1)
                            self.neg_candidate_item[start_idx:start_idx + len_seq + 1] = b[
                                i][0][:len_seq+1]
                            # else:
                            #     start_idx = random.randint(0, (len_seq%self.max_neg_item) - self.max_neg_item - 1)
                            #     self.neg_candidate_item[start_idx:start_idx + len_seq + 1] = b[
                            #         i][0]
                            #     self.neg_candidate_item[self.max_neg_item//2:]=b[i][0][:self.max_neg_item//2]

                        if len(self.neg_candidate_cat) < self.max_neg_cat:
                            self.neg_candidate_cat.extend(b[i][1])
                            if len(self.neg_candidate_cat) > self.max_neg_cat:
                                self.neg_candidate_cat = self.neg_candidate_cat[
                                    0:self.max_neg_cat]
                        else:
                            len_seq = len(b[i][1])
                            if self.max_neg_cat < len_seq:
                                len_seq = len_seq % self.max_neg_cat
                            # print("----self.max_neg_cat", self.max_neg_cat)
                            # print("-----len_seq", len_seq)
                            start_idx = random.randint(0, self.max_neg_cat - len_seq - 1)
                            self.neg_candidate_item[start_idx:start_idx + len_seq + 1] = b[
                                i][1]
                        for _ in range(len(item)):
                            neg_item[i].append(self.neg_candidate_item[random.randint(
                                0, len(self.neg_candidate_item) - 1)])
                        for _ in range(len(cat)):
                            neg_cat[i].append(self.neg_candidate_cat[random.randint(
                                0, len(self.neg_candidate_cat) - 1)])

                    for i in range(len(b)):
                        res = []
                        # b1 = np.array(item[i]).astype('int64')
                        # b2 = np.array(cat[i]).astype('int64')
                        # a1=paddle.to_tensor(b1)
                        # a2=paddle.to_tensor(b2)
                        # res.append(a1)
                        # res.append(a2)
                        res.append(np.array(item[i]).astype('int64'))
                        res.append(np.array(cat[i]).astype('int64'))    

                        flag = int(b[i][4])
                        # tar_item = int(b[i][2])
                        # sam_item = int(neg_item[i][-1])
                        if flag ==0:
                            tar_item = int(b[i][2])

                        else:
                            tar_item = int(neg_item[i][-1])
                            a=int(b[i][2])
                            neg_item[i][-1]=str(a)

                        res.append(np.array([tar_item]))
                        # res.append(np.array(user_seq))
                        # print("---------tar_item -->",tar_item)
                        path_set = self.graph_index.get_path_of_item(tar_item)
                        # print("---------path_set -->",path_set)
                        item_path_kd_label = []
                        for path in path_set[0]:
                            path_label = np.array(self.graph_index.path_id_to_kd_represent(path))
                            item_path_kd_label.append(path_label)
                        #     print("--------kd_path_label------>",path_label)
                        # print("--------list_kd_label------>",item_path_kd_label)

                        res.append(np.array(item_path_kd_label).astype("int64"))

                        if self.use_multi_task_learning:
                            mul_label = [tar_item]
                            res.append(np.array(mul_label).astype('int64'))
                            res.append(np.array(neg_item[i]).astype('int64'))

                            # res.append(np.array(neg_item[i]).astype('int64'))
                        res.append(mask)

                        # res.append(np.array(mask).astype('float'))
                        # print("data[0]]",res[0].shape)
                        # print("data-mask",res[-1].shape)
                        # print("---------res",res)
                        yield res

        len_bg = len(bg)
        if len_bg != 0:
            sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
            bg = []
            remain = len_bg % batch_size
            for i in range(0, len_bg - remain, batch_size):
                b = sortb[i:i + batch_size]

                max_len = max(len(x[0]) for x in b)

                itemInput = [x[0] for x in b]
                itemRes0 = np.array(
                    [x + [0] * (max_len - len(x)) for x in itemInput])
                item = itemRes0.astype("int64").reshape([-1, max_len])
                # item = [x[0] for x in b]
                catInput = [x[1] for x in b]
                catRes0 = np.array(
                    [x + [0] * (max_len - len(x)) for x in catInput])
                cat = catRes0.astype("int64").reshape([-1, max_len])
                # cat = [x[1] for x in b]

                len_array = [len(x[0]) for x in b]
                mask = np.array(
                    [[0] * x + [-1e9] * (max_len - x)
                     for x in len_array]).reshape([-1, max_len, 1])
                target_item_seq = np.array(
                    [[x[2]] * max_len for x in b]).astype("int64").reshape(
                        [-1, max_len])
                target_cat_seq = np.array(
                    [[x[3]] * max_len for x in b]).astype("int64").reshape(
                        [-1, max_len])
                neg_item = [None] * len(item)
                neg_cat = [None] * len(cat)

                for i in range(len(b)):
                    neg_item[i] = []
                    neg_cat[i] = []
                    if len(self.neg_candidate_item) < self.max_neg_item:
                        self.neg_candidate_item.extend(b[i][0])
                        if len(self.neg_candidate_item) > self.max_neg_item:
                            self.neg_candidate_item = self.neg_candidate_item[
                                0:self.max_neg_item]
                    else:
                        len_seq = len(b[i][0])
                        start_idx = random.randint(0, self.max_neg_item - len_seq - 1)
                        self.neg_candidate_item[start_idx:start_idx + len_seq + 1] = b[
                            i][0]

                    if len(self.neg_candidate_cat) < self.max_neg_cat:
                        self.neg_candidate_cat.extend(b[i][1])
                        if len(self.neg_candidate_cat) > self.max_neg_cat:
                            self.neg_candidate_cat = self.neg_candidate_cat[
                                0:self.max_neg_cat]
                    else:
                        len_seq = len(b[i][1])
                        start_idx = random.randint(0, self.max_neg_cat - len_seq - 1)
                        self.neg_candidate_item[start_idx:start_idx + len_seq + 1] = b[
                            i][1]
                    for _ in range(len(item)):
                        neg_item[i].append(self.neg_candidate_item[random.randint(
                            0, len(self.neg_candidate_item) - 1)])
                    for _ in range(len(cat)):
                        neg_cat[i].append(self.neg_candidate_cat[random.randint(
                            0, len(self.neg_candidate_cat) - 1)])                        

                for i in range(len(b)):
                    res = []
                    # b1 = np.array(item[i]).astype('int64')
                    # b2 = np.array(cat[i]).astype('int64')
                    # a1=paddle.to_tensor(b1)
                    # a2=paddle.to_tensor(b2)
                    # res.append(a1)
                    # res.append(a2)
                    # user_seq = paddle.concat([a1,a2], axis=-1)
                    res.append(np.array(item[i]).astype('int64'))
                    res.append(np.array(cat[i]).astype('int64'))                    
                    flag = int(b[i][4])
                    # tar_item = int(b[i][2])
                    # sam_item = int(neg_item[i][-1])
                    if flag ==0:
                        tar_item = int(b[i][2])

                    else:
                        tar_item = int(neg_item[i][-1])
                        a=int(b[i][2])
                        neg_item[i][-1]=str(a)

                    res.append(np.array([tar_item]))
                    # res.append(np.array(user_seq))
                    path_set = self.graph_index.get_path_of_item(tar_item)
                    #print("path_set -->",path_set)
                    item_path_kd_label = []
                    for path in path_set[0]:
                        path_label = np.array(self.graph_index.path_id_to_kd_represent(path))
                        item_path_kd_label.append(path_label)
                    res.append(np.array(item_path_kd_label).astype("int64"))

                    if self.use_multi_task_learning:
                        mul_label = [tar_item]
                        res.append(np.array(mul_label).astype('int64'))
                        res.append(np.array(neg_item[i]).astype('int64'))

                        # res.append(np.array(neg_item[i]).astype('int64'))

                    # print("---------res",res)
                    res.append(mask)
                    # res.append(np.array(mask).astype('float'))
                    # print("data[0]]",res[0].shape)
                    # print("data-mask",res[-1].shape)
                    yield res
