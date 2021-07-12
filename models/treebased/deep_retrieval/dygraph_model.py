# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import os
import paddle.nn as nn
import paddle.nn.functional as F
import math
from operator import itemgetter, attrgetter

from net import DeepRetrieval
from paddle.distributed.fleet.dataset.index_dataset import GraphIndex


class DygraphModel():
    # define model
    def save_item_path(self, path, prefix):
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, str(prefix))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, self.path_save_file_name)
        self.graph_index._graph.save(path)     

    def load_item_path(self, path):
        path = os.path.join(path, self.path_save_file_name)
        self.graph_index._graph.load(path)

    def create_model(self, config):
        # print("create model")
        self.path_save_file_name = "path_save";
        self.width = config.get("hyper_parameters.width")
        self.height = config.get("hyper_parameters.height")
        self.beam_search_num = config.get(
            "hyper_parameters.beam_search_num")
        self.item_path_volume = config.get(
            "hyper_parameters.item_path_volume")
        self.user_embedding_size = config.get(
            "hyper_parameters.user_embedding_size")
        self.graph_index = GraphIndex(
            "test", self.width, self.height, self.item_path_volume)

        init_model_path = config.get("model_init_path")

        if init_model_path == None:
            self.graph_index._init_by_random()
        else:
            self.graph_index._init_graph(os.path.join(init_model_path, self.path_save_file_name))
        self.use_multi_task_learning = config.get("hyper_parameters.use_multi_task_learning")
        self.item_count = config.get("hyper_parameters.item_count")
        if self.use_multi_task_learning:
            self.multi_task_layer_size = config.get("hyper_parameters.multi_task_layer_size")
        else:
            self.multi_task_layer_size = None

        self.data_format = config.get("hyper_parameters.data_format")
        self.item_emb_size = config.get("hyper_parameters.item_emb_size")
        self.cat_emb_size = config.get("hyper_parameters.cat_emb_size")
        self.cat_count = config.get("hyper_parameters.cat_count")
        self.item_count = config.get("hyper_parameters.item_count")
        self.is_sparse = config.get("hyper_parameters.is_sparse")
        self.recall_num = config.get("hyper_parameters.recall_num")

        dr_model = DeepRetrieval(self.width, self.height, self.beam_search_num,
                                 self.item_path_volume, self.user_embedding_size, self.item_count, self.data_format, self.item_emb_size,self.cat_emb_size,self.cat_count, 
                                 self.is_sparse, self.use_multi_task_learning, self.multi_task_layer_size, is_static=False)

        # print("--------dr_model-----------",dr_model)
        return dr_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        # print("--------batch_data-----------",batch_data)

        #print("batch_data ",batch_data, "len_batch_data",len(batch_data))
        # user_embedding = paddle.to_tensor(batch_data[0].numpy().astype(
        #     'float32').reshape(-1, self.user_embedding_size))

        # item_id = batch_data[1]  # (batch_size, 1)   if use_multi_task (batch_size,3)
        # item_id = item_id.numpy().tolist()
        # item_path_id = []
        # multi_task_pos_label = []
        # multi_task_neg_label = []
        # for i in item_id:
        #     item_path_id.append(self.graph_index.get_path_of_item(i[0]))
        #     if self.use_multi_task_learning:
        #         multi_task_pos_label.append([i[0]])
        #         multi_task_neg_label.append([i[1]])


        # item_path_kd_label = []

        # # for every example
        # for item_path in item_path_id:
        #     item_kd_represent = []
        #     # for every path choice of item
        #     for item_path_j in item_path[0]:
        #         path_label = np.array(
        #             self.graph_index.path_id_to_kd_represent(item_path_j))
        #         #item_kd_represent.append(paddle.to_tensor(
        #         #    path_label.astype('int64').reshape(1, self.width)))
        #         item_kd_represent.append(paddle.to_tensor(
        #            path_label.astype('int32').reshape(self.width)))
        #     item_path_kd_label.append(paddle.reshape(paddle.concat(item_kd_represent,axis=-1),[-1, self.width]))
        # #print("label shape ",item_path_kd_label.shape)    
        # item_path_kd_label=paddle.concat(item_path_kd_label, axis = 0)

        if self.data_format == "amazon_book_behavior":
            if self.use_multi_task_learning:
                item_seq, cat_seq, item_id, item_path_kd_label, mul_label, neg_label, mask = batch_data[0:7]
                return item_seq, cat_seq, item_id, None, item_path_kd_label, mul_label, neg_label, mask
            else:
                item_seq, cat_seq, item_id, item_path_kd_label = batch_data[0:4]
                mask = batch_data[-1]
                return item_seq, cat_seq, item_id, None, item_path_kd_label, None, None, mask

        elif self.data_format == "user_embedding":
            if self.use_multi_task_learning:
                item_id, user_embedding, item_path_kd_label, mul_label, neg_label = batch_data[0:5]
                return None, None, item_id, user_embedding, item_path_kd_label, mul_label, neg_label, None
            else:
                item_id, user_embedding, item_path_kd_label = batch_data[0:3]
                return None, None, item_id, user_embedding, item_path_kd_label, None, None, None
        # return batch[0:2]
        #     #return user_embedding, item_path_kd_label, multi_task_pos_label ,   multi_task_neg_label
        #     multi_task_pos_label = paddle.to_tensor(multi_task_pos_label)
        #     multi_task_neg_label = paddle.to_tensor(multi_task_neg_label)
        # item_id, user_embedding, item_path_kd_label = batch_data[0:3]
        # #return user_embedding, item_path_kd_label, multi_task_pos_label ,   multi_task_neg_label
        # return  item_id, user_embedding, item_path_kd_label, None, None

    def create_infer_feeds(self, batch_data, config):
        if self.data_format == "user_embedding":
            user_embedding = paddle.to_tensor(batch_data[1].numpy().astype(
                'float32').reshape(-1, self.user_embedding_size))
            # print("----batch_data[0]", batch_data[0])
            items = paddle.reshape(batch_data[0], [-1])
            # print("----items", items)
            item_Id = batch_data[0].numpy().astype('int64')
            # print("---item_Id",item_Id)
            return user_embedding, item_Id
        elif self.data_format == "amazon_book_behavior":
            user_seq = batch_data[0].numpy().astype('int64')
            user_cat = batch_data[1].numpy().astype('int64')
            label_items = batch_data[2].numpy().astype('int64')
            # print("----batch_data---", batch_data)
            # print("----label_items--", label_items)
            return user_seq, user_cat, label_items

            # item_seq, cat_seq = batch_data[0],batch_data[1]
            # # print("----batch_data[2]", batch_data[2])
            # items = paddle.reshape(batch_data[2], [-1])
            # tar_items = batch_data[0][(len(batch_data[0])//2):].expand(batch_data[2])
            # items = paddle.reshape(tar_items, [-1])
            # print("----items", items)
            # item_Ids = items.numpy().astype('int64')
            # print("---item_Id",item_Ids)
            # return item_seq, cat_seq, item_Ids

    # define loss function by predicts and label
    def create_loss(self, path_prob, multi_task_loss):
        # path_prob: (batch_size * J, D)
        path_prob = paddle.prod(
            path_prob, axis=1, keepdim=True)  # (batch_size* J, 1)
        item_path_prob = paddle.reshape(
            path_prob, (-1, self.item_path_volume))  # (batch_size, J)
        item_path_prob = paddle.sum(item_path_prob, axis=1, keepdim=True)
        # print("-----item_path_prob",item_path_prob)
        epsilon=1e-9
        item_path_prob_log = paddle.log(item_path_prob+epsilon)   # epsilon;
        
        # print("-----item_path_prob_log",item_path_prob_log)
        # print("-----multi_task_loss",multi_task_loss)
        # print("multi_task_loss: {}".format(multi_task_loss))
        cost_dr = -1 * paddle.sum(item_path_prob_log)
        # print("dr_cost: {}".format(cost))
        if self.use_multi_task_learning:
            cost = cost_dr + multi_task_loss
        # print("cost: {}".format(cost))
        return cost
        # return cost_dr
        # return multi_task_loss

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name
    # def create_metrics(self):
    #     metrics_list_name = ["auc"]
    #     auc_metric = paddle.metric.Auc("ROC")
    #     metrics_list = [auc_metric]
    #     return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        # print("------------dynamic_____train_forward------------")
        
        item_seq, cat_seq, item_id, user_embedding, item_path_kd_label,multi_task_pos_label,multi_task_neg_label, mask = self.create_feeds(batch_data, config)

        # print("--------item_seq, cat_seq, item_id, user_embedding,",item_seq, cat_seq, item_id, user_embedding,)

        path_prob,multi_task_loss = dy_model.forward(
            item_seq, cat_seq, user_embedding,item_path_kd_label,multi_task_pos_label,multi_task_neg_label, mask, False, False)

        loss = self.create_loss(path_prob,multi_task_loss)
        # # update metrics
        # predict_2d = paddle.concat(x=[1 - path_prob, path_prob], axis=1)
        # metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        print_dict = {'loss': loss}
        # print("------loss",loss)
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, batch_data, config):
        # dr_mlp --> path and pro
        # item_ids = label_items
        if self.data_format == "user_embedding":
            user_embedding, item_ids = self.create_infer_feeds(batch_data,config)
            # user_embedding = self.create_infer_feeds(batch_data,config)
            kd_path, path_prob = dy_model.forward(user_embedding, is_infer=True)
        elif self.data_format == "amazon_book_behavior":
            user_seq, user_cat, item_ids = self.create_infer_feeds(batch_data,config)
            kd_path, path_prob, user_embedding = dy_model.forward(user_seq, user_cat, is_infer=True)
        # path to items --> recall metrics
        dy_model.init_metrics()
        label_batchSize = item_ids 
        # print("------label_batchSize", label_batchSize)
        # print("kd_path: {}".format(kd_path))
        kd_path_list = kd_path.numpy().astype(np.int32).tolist()
        # print("kd_path_list: {}".format(kd_path_list))
        # print("path_prob: {}".format(path_prob))
        item_list = []
        em_dict = {}
        # item = 0
        em_dict[0] = []
        user_recalls = []
        item_recalls = []
        total_recall_len = 0
        recall_lens = []
        label_items = []

# emb, kd_path, label in zip(batch_emb, path_list, batch_items):
        for batch_kdpath, batch_label, batch_user in zip(kd_path_list, label_batchSize, user_embedding):
            # print("------label_batchSize", label_batchSize)
            # print("-----batch_label", batch_label)
            batch_item_list = []
            # print("batch: {}".format(batch))
            path_batch=[]
            # beamSize paths for a batch user
            for path_idx, path in enumerate(batch_kdpath):
                # print("path0:", path)
                # path_id = self.graph_index.kd_represent_to_path_id(path)
                # print("path1:",path_id)
                # path_item_ids = self.graph_index.get_item_of_path(path_id)
                # print("get_item_of_path----path_item_ids:",path_item_ids)  # ('path_item_ids:', [[]])
                path_batch.append(self.graph_index.kd_represent_to_path_id(path))
            #     print("---path_batch", path_batch)
            # print("--all-path_batch", path_batch)
            # recall for one (a batch data) user
            batch_recall_items = self.graph_index._graph.gather_unique_items_of_paths(path_batch)
            # print("---all-gather_unique_items_of_paths", recall_items)

            # if recall for a certain user is under recall_bar, get the metric for the user and go on with the next user(or batch)
            if len(batch_recall_items) < self.recall_num:
                # print("recall_items ", recall_items)
                # print("label ", label)
                # print("-----batch_label", batch_label)
                dy_model.calculate_metric(batch_recall_items, batch_label)
                continue

            # when recall_num over bar, go on with rerank
            # under recallBar with same recall loop, single batchSize, dif users, single userEmb
            # recall for a certain same batch user, if recall_mnt over bar, rerank items for this user. 
            # if the recall_mnt under the recall_bar, go on with the next batch(or next user)
            user_recalls = user_recalls + ([batch_user] * len(batch_recall_items))
            total_recall_len = total_recall_len + len(batch_recall_items)
            recall_lens.append(len(batch_recall_items))
            item_recalls = item_recalls + batch_recall_items
            label_items.append(batch_label)
            # recall_bar is set for the same user(a batch);
            # total_recall_len is for all users in the same batchSize, if get the recall_bar, rerank them together.
            if total_recall_len >=1: # why ???????????????????
                user_recalls_np = np.array(user_recalls).astype("float32")
                item_recalls_np = np.array(item_recalls).astype("int64")
                # print("---user_recalls_np", user_recalls_np)
                # print("---item_recalls_np", item_recalls_np)
                user_recalls_emb = paddle.to_tensor(user_recalls_np, dtype="float32")
                item_recalls_emb = paddle.to_tensor(item_recalls_np, dtype="int64")
                # print("---user_recalls_emb", user_recalls_emb)
                # print("---item_recalls_emb", item_recalls_emb)
                
                rerank_outs = dy_model.rerank(user_recalls_emb, item_recalls_emb)
                # outs = rerank(....)
                rerank_score = np.array(rerank_outs[0])
                cur_len=0
                # ith batch user
                for i in range(len(recall_lens)):
                    if recall_lens[i] == 0:
                        dy_model.calculate_metric([0], label_items[i])
                    else:
                        rerank_item_pro = [(x,y) for x,y in zip(rerank_score[cur_len: cur_len+recall_lens[i]],
                                                item_recalls[cur_len: cur_len+recall_lens[i]])]
                        rerank_item_pro = sorted(rerank_item_pro, key=itemgetter(0), reverse=True)[0: self.recall_num]
                        rerank_item = [int(item_pro[1]) for item_pro in rerank_item_pro]
                        # if recall nums from drNet over recall bar, go on with rank in net. 
                        # recall is the metric for the terminal items we get 
                        dy_model.calculate_metric(rerank_item, label_items[i])
                    cur_len = cur_len + recall_lens[i]
                total_recall_len = 0
                cur_len = 0
                item_recalls = []
                recall_lens = []
                user_recalls = []
                label_items = []
        # update metric with a certain batch-user, and get the final metric updated by all batchSize-users.
        user_embedding = []
        precision, recall, F1 = dy_model.final_metrics()
        return path_prob, precision, recall, F1

        # item_id

        #     for item in path_item_ids[0]:
        #         batch_item_list.append(item)
        #     print("batch_item_list:", batch_item_list)
        #     item_list.append(batch_item_list)
        # print("item_list",item_list)
        #         em_dict[0].append("{}:{}".format(path_id, prob))

       # multi_task net --> recall_items dot userEmb

        # re_rec_item, re_rec_index = dy_model.forward(user_embedding=user_embedding, bs_item_list=item_list, re_infer=True) # all paths for the user_embedding
        # return metrics_list, None