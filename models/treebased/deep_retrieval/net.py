# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import os
import paddle.fluid as fluid
import numpy as np
from paddle.vision.transforms import functional as T


class DeepRetrieval(nn.Layer):

    def expand_layer(self, input, n):
        # print("-----expand_layer_input", input)
        # print("-----expand_n", n)
        # expand input (batch_size, shape) -> (batch_size * n, shape)
        col_size = input.shape[1]
        arr = [input] * n

        input = paddle.concat(arr, axis=1)
        # print("-----concat_layer_out", input)

        input = paddle.reshape(input, [-1, col_size])
        # print("-----expand_layer_out", input)

        return input

    def rerank(self, user_emb, item_ids):
        temp = user_emb
        # (batch, dot_product_size)
        for i in range(len(self.multi_task_mlp_layers)):
            temp = self.multi_task_mlp_layers[i](temp)     
        item_emb = self.multi_task_item_embedding(item_ids)
        return paddle.dot(temp, item_emb)

    def init_metrics(self):
        self.presision = 0
        self.count = 0
        self.recall = 0

    def upate_metrics(self, presision_value, recall_value):
        self.count = self.count + 1
        self.presision = self.presision + presision_value
        self.recall = self.recall + recall_value

    def final_metrics(self):
        res_presision = self.presision/self.count
        res_recall = self.recall/self.count
        flag=res_presision+res_recall
        if flag == 0:
            res_F1 = None
        else:
            res_F1 = 2*res_presision*res_recall/(res_presision+res_recall)
        return res_presision, res_recall, res_F1
        # return self.presision/self.count, self.recall/self.count,  

    def calculate_metric(self,recall_list,user_items_list):
        if len(recall_list) == 0:
            recall_list = [0]
        # print("recall_list",recall_list)
        # print("user_items_list",user_items_list)
        user_dict = set(user_items_list)
        recall_dict = set(recall_list)
        common_len = len(user_dict & recall_dict)
        self.upate_metrics(common_len/len(recall_dict), common_len/len(user_dict))

    def __init__(self, width, height, beam_search_num, item_path_volume, user_embedding_size, item_count, data_format,item_emb_size,cat_emb_size,cat_count, is_sparse,
                 use_multi_task_learning=True, multi_task_mlp_size=None, is_static=False):
        super(DeepRetrieval, self).__init__()
        self.width = width
        self.height = height
        self.beam_search_num = beam_search_num
        self.item_path_volume = item_path_volume

        self.user_embedding_size = user_embedding_size
        self.data_format = data_format
        self.is_sparse = is_sparse
        self.item_count = item_count
        self.item_emb_size = item_emb_size
        self.cat_emb_size =cat_emb_size
        self.cat_count = cat_count

        # if self.data_format =="amazon_book_behavior":
        self.cat_emb_size = cat_emb_size
        self.item_emb_size = item_emb_size
        self.gru_input_size = self.cat_emb_size + self.item_emb_size 
        
        self.hist_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            sparse=self.is_sparse,
            padding_idx=0,
            name="item_emb")

        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            sparse=self.is_sparse,
            padding_idx=0,
            name="cat_emb")


        in_sizes = [user_embedding_size + i *
                    user_embedding_size for i in range(self.width)]
        # print("in_sizes: {}".format(in_sizes))
        out_sizes = [self.height] * self.width
        # print("out_sizes: {}".format(out_sizes))

        self.use_multi_task_learning = use_multi_task_learning
        self.mlp_layers = []
        self.multi_task_mlp_layers_size = [user_embedding_size]
        self.multi_task_mlp_layers = []

        # mlp_layers: [ [in_sizes[0],out_sizes[0]], [in_sizes[1], out_sizes[1]], [in_sizes[2], out_sizes[2]] ]
        for i in range(width):
            linear = paddle.nn.Linear(
                in_features=in_sizes[i],
                out_features=out_sizes[i],
                weight_attr=paddle.ParamAttr(
                    name="C_{}_mlp_weight".format(i),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(out_sizes[i]))))
            self.mlp_layers.append(linear)
            self.add_sublayer("C_{}_mlp_weight".format(i), linear)

        self.path_embedding = paddle.nn.Embedding(
            self.height,
            self.user_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="path_embedding",
                initializer=paddle.nn.initializer.Uniform())
        )

        if self.use_multi_task_learning:
            self.item_count = item_count
            for i in multi_task_mlp_size:
                self.multi_task_mlp_layers_size.append(i)
            for i in range(len(self.multi_task_mlp_layers_size) - 1):
                linear = paddle.nn.Linear(
                    in_features=self.multi_task_mlp_layers_size[i],
                    out_features=self.multi_task_mlp_layers_size[i + 1],
                    weight_attr=paddle.ParamAttr(
                        name="multi_task_{}_mlp_weight".format(i),
                        initializer=paddle.nn.initializer.Normal(
                            std=1.0 / math.sqrt(out_sizes[i]))))
                self.multi_task_mlp_layers.append(linear)
                self.add_sublayer("multi_task_{}_mlp_weight".format(i), linear)
            self.dot_product_size = self.multi_task_mlp_layers_size[-1]
            self.multi_task_item_embedding = paddle.nn.Embedding(
                self.item_count,
                self.dot_product_size,
                weight_attr=paddle.ParamAttr(
                    name="multi_task_item_embedding_weight",
                    initializer=paddle.nn.initializer.Uniform()))
        if self.data_format == "amazon_book_behavior":
            self.gru=paddle.nn.GRU(self.gru_input_size, self.user_embedding_size, 2)  # where to define user_dim

        if is_static:
            self.em_startup_program = fluid.default_startup_program().clone()
            self.em_main_program = paddle.static.default_main_program().clone()            



    def generate_candidate_path_for_item(self, input_embeddings, beam_size, is_static):
        # print("ori_input_embeddings:", input_embeddings) # [[],[]]
        # print("beam_size:",beam_size)
        if self.data_format =="user_embedding":
            input_embeddings = input_embeddings   # shape=[2, 2]
            # print("hhhhh----input_embeddings",input_embeddings)
        elif self.data_format =="amazon_book_behavior":
            if is_static == False:
                # for user_seq in input_embeddings:
                #     item_seq=user_seq[0]
                #     cat_seq=user_seq[-1]
                #     item_emb = paddle.to_tensor(np.array(item_seq).astype('float32'))
                #     cat_emb = paddle.to_tensor(np.array(cat_emb).astype('float32'))
                # print("--------gen-----input_embeddings-----",input_embeddings)
                hist_cat_seq = input_embeddings[-1]
                hist_item_seq = input_embeddings[0]
                # print("---------gen-----hist_item_seq",hist_item_seq)
                # print("---------gen-----hist_cat_seq",hist_cat_seq)
                user_embeddings=[]

                for item,cat in zip(hist_item_seq,hist_cat_seq):
                    # print("item",item)
                    item_emb_tensor = paddle.to_tensor(np.array(item))
                    # item_emb_tensor = paddle.to_tensor(np.array(item).astype('float32'))
                    cat_emb_tensor = paddle.to_tensor(np.array(cat)) #T.to_tensor(np.array(cat).astype('float32'))

                # item_emb_tensor = paddle.to_tensor(np.array(hist_item_seq).astype('int64'))
                # cat_emb_tensor = paddle.to_tensor(np.array(hist_cat_seq).astype('int64'))
                    # print("---------gen-----item_emb",item_emb_tensor)
                    # print("---------gen-----cat_emb",cat_emb_tensor)

                    # hist_item_emb = self.hist_item_emb_attr(item)
                    # hist_cat_emb = self.hist_cat_emb_attr(cat)

                    hist_item_emb = self.hist_item_emb_attr(item_emb_tensor)
                    hist_cat_emb = self.hist_cat_emb_attr(cat_emb_tensor)
                    # print("hist_item_emb-------:",hist_item_emb) 
                    # print("hist_cat_emb-------:",hist_cat_emb) 

                    hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=1)
                    # print("hist_seq_concat========",hist_seq_concat)
                    embSize = self.cat_emb_size + self.item_emb_size
                    # print("embSize======",embSize)
                    gru_input = paddle.reshape(hist_seq_concat, [1,-1,128])
                    
                    # print("gru_input-------:",gru_input) 
                    # print("---------gen-----self.gru",self.gru)
                    # gru_out, cur_status = self.gru(gru_input,gru_init)   # tensor [1, 2, 2]; tensor [1, 1, 2]
                    gru_out, cur_status = self.gru(gru_input)   # tensor [1, 2, 2]; tensor [1, 1, 2]
                    # print("cur_status",cur_status)
                    # user_embedding_status = paddle.sum(cur_status, axis=0)
                    # print("user_embedding_status", user_embedding_status)
                    # print("---------gru_out", gru_out)
                    user_embedding = paddle.sum(gru_out, axis=1)
                    # print("---------gen-----user_embedding", user_embedding)
                    user_embeddings.append(user_embedding)

                    # print("---------user_embedding---------", user_embedding)
                input_embeddings = paddle.to_tensor(user_embeddings)
                # print("---------input_embeddings---------", input_embeddings)
                input_embeddings = paddle.reshape(input_embeddings,[-1,embSize])
                # print("---------input_embeddings---------", input_embeddings)

            elif is_static == True:
                # print("======net====generate===item-paths==input_embeddings", input_embeddings)
                cat_emb_tensors = input_embeddings[-1]
                item_emb_tensors = input_embeddings[0]
                # print("======net====generate===item-paths==input_embeddings[0]", item_emb_tensors)
                # paddle.static.Print(input_embeddings)
                # paddle.static.Print(cat_emb_tensors)
                # paddle.static.Print(item_emb_tensors)
                # print("------input_embeddings",input_embeddings)
                # print("-------cat_emb_tensors",cat_emb_tensors)
                # print("-------item_emb_tensors",item_emb_tensors)
                user_embeddings=[]

                hist_item_emb = self.hist_item_emb_attr(item_emb_tensors)
                hist_cat_emb = self.hist_cat_emb_attr(cat_emb_tensors)
                hist_emb_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=1)
                # print("====hist_item_emb=====",hist_item_emb)
                # print("hist_emb_concat========",hist_emb_concat)
                embSize = self.cat_emb_size + self.item_emb_size
                # print("embSize======",embSize)
                gru_input = paddle.reshape(hist_seq_concat, [1,-1,128])
                gru_out, cur_status = self.gru(gru_input)   # tensor [1, 2, 2]; tensor [1, 1, 2]
                # print("=======gru_out",gru_out)
                user_embedding = paddle.sum(gru_out, axis=1)
                # print("======user_embedding", user_embedding)
                # paddle.static.Print(user_embedding)

                
                # for item_emb_tensor,cat_emb_tensor in zip(item_emb_tensors,cat_emb_tensors):
                #     hist_item_emb = self.hist_item_emb_attr(item_emb_tensor)
                #     hist_cat_emb = self.hist_cat_emb_attr(cat_emb_tensor)
                #     hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=1)
                #     # print("hist_seq_concat========",hist_seq_concat)
                #     embSize = self.cat_emb_size + self.item_emb_size
                #     # print("embSize======",embSize)
                #     gru_input = paddle.reshape(hist_seq_concat, [1,-1,128])
                #     gru_out, cur_status = self.gru(gru_input)   # tensor [1, 2, 2]; tensor [1, 1, 2]
                #     user_embedding = paddle.sum(gru_out, axis=1)
                #     user_embeddings.append(user_embedding)
                # input_embeddings = paddle.concat(user_embeddings,axis=1)
                input_embeddings = paddle.reshape(input_embedding,[-1,embSize])

        # print("beam_size, self.height", beam_size, self.height)
        # print("input_embeddings:", input_embeddings)
        if beam_size > self.height:
            beam_size = self.height
        height = paddle.full(
            shape=[1, 1], fill_value=self.height, dtype='int64')
        prob_list = []
        saved_path = None
        w = []
        row = paddle.zeros_like(input_embeddings, dtype="int64")
        # print("row_zeros:", row)
        row = paddle.sum(row, axis=-1)
        # print("row_sum:", row)
        # [batch,1]
        row = paddle.reshape(row, [-1, 1])
        # print("row_reshape:", row)

        for i in range(beam_size):
            x = row + i
            w.append(x)
        #     print("ith_beam row + i ; w", x, w)
        # print("beam + row ; w", x, w)
        # [batch] all zeros
        batch_row = paddle.reshape(row, [-1])
        # print("batch_(reshape)row:", batch_row)
        row = paddle.concat(w, axis=-1)
        # print("concatw_row:",row)
        # row = [0,1,2...beam-1,0,1,2....] ,shape = [beam *batch]
        row = paddle.reshape(row, [-1])
        # print("reshape_row",row)
        for i in range(self.width):
            if i == 0:
                # print("------firstLayer--------")
                # [batch, height]
                pro = F.softmax(self.mlp_layers[0](input_embeddings))
                # print("self.mlp_layers[0]:", self.mlp_layers[0])
                # print("pro:", pro)
                # [height]
                pro_sum = paddle.sum(pro, axis=0)
                # print("pro_sum:", pro_sum)
                # [beam_size],[beam_size]
                _, index = paddle.topk(pro_sum, beam_size)
                # print("topK_index:", index)
                # [1, beam_size]
                saved_path = paddle.unsqueeze(index, 0)
                # print("topK_saved_path:", saved_path)

                # [batch,height] -> [height, batch] -> [beam,batch]
                #last_prob = paddle.index_select(paddle.reshape(pro, [self.height, -1]), index)
                last_prob = paddle.index_select(paddle.transpose(pro, [1,0]), index)
                # print("last_prob(transpose pro & index_select):", last_prob)
                # [batch, beam]
                #last_prob = paddle.reshape(last_prob, [-1, beam_size])
                last_prob = paddle.transpose(last_prob,[1,0])
                # print("last_prob transpose:", last_prob)
                #prob_list.append(last_prob)
                # [batch * beam, emb_size]
                input_embeddings = self.expand_layer(input_embeddings, beam_size)
                # print("inputEmb_expand_beamSize",input_embeddings)
                # # [batch, 1, emb_size]
                # input_embeddings = paddle.unsqueeze(input_embeddings, 1)
                # # [batch,beam,emb_size]
                input_embeddings = paddle.reshape(input_embeddings, [-1, beam_size, self.user_embedding_size])
                # print("input_embeddings_reshape:", input_embeddings)
            else:
                # print("------conLayers--------")
                # [i,beam] ->[beam, i]
                reverse_saved_path = paddle.transpose(saved_path, [1, 0])
                # print("saved_path_transpose:", reverse_saved_path)
                # [beam, i,emb_size ]
                saved_path_emb = self.path_embedding(reverse_saved_path)
                # print("saved_path_emb:",saved_path_emb)
                # [beam, i * emb_size ]
                input = paddle.reshape(saved_path_emb, [beam_size, -1])
                # print("input_path_emb_reshape",input)
                # [beam * batch, i * emb_size]
                input = paddle.index_select(input, row)
                # print("input_select:",input)
                #print("input shape ", input.shape)
                # [batch, beam, i * emb_size]
                input = paddle.reshape(input, [-1, beam_size, i * self.user_embedding_size])
                # print("input_reshape", input)
                # # input = paddle.concat(emb_list,axis=-1)
                # input = paddle.unsqueeze(input, 0)
                # # [batch, beam, i * emb_size]
                # input = paddle.expand(input, [batch_size, beam_size, i * self.user_embedding_size])

                # [batch, beam, (i+1) * emb_size]
                input = paddle.concat([input_embeddings, input], axis=-1)
                # print("concat_input",input)
                # [batch, beam_size, height]
                out = F.softmax(self.mlp_layers[i](input))
                # print("softmax_pro",out)

                # [batch, beam] -> [batch * height, beam]
                extend_pro = self.expand_layer(last_prob,self.height)
                # print("last_prob_expend_height:", extend_pro)

                # [batch * height, beam] -> [batch, height, beam]
                extend_pro = paddle.reshape(extend_pro, [-1, self.height, beam_size])
                # print("reshape_extend_pro",extend_pro)
                # [batch, beam, height]
                extend_pro = paddle.transpose(extend_pro,[0,2,1])
                # print("extend_pro",extend_pro)

                # [batch, beam, height]
                temp_prob = paddle.multiply(extend_pro, out)
                # print("temp_prob", temp_prob)
                # [beam, height]
                pro_sum = paddle.sum(temp_prob, axis=0)
                # print("pro_sum",pro_sum)
                # [beam * height]
                pro_sum = paddle.reshape(pro_sum, [-1])
                # print("pro_sum",pro_sum)
                # [beam]
                _, index = paddle.topk(pro_sum, beam_size)
                # print("index",index)
                # [1,beam]
                beam_index = paddle.floor_divide(index, height)
                # print("beam_index",beam_index)
                item_index = paddle.mod(index, height)
                # print("item_index",item_index)
                # [batch, beam, height] to be checked
                temp_prob = paddle.index_select(temp_prob,paddle.reshape(beam_index,[-1]),axis=1)
                # print("temp_prob",temp_prob)
                # [batch * beam, height]
                temp_prob = paddle.reshape(temp_prob, [-1, self.height])
                # print("re_temp_prob",temp_prob)
                # [batch,beam]
                batch_item_index = paddle.index_select(item_index, batch_row)
                # print("batch_item_index",batch_item_index)
                # [batch,beam] -> [batch * beam,1]
                batch_item_index = paddle.reshape(batch_item_index, [-1,1])
                # print("re_batch_item_index",batch_item_index)

                # [batch * beam, 1]
                last_prob = paddle.index_sample(temp_prob, batch_item_index)
                # print("last_prob",last_prob)

                # [batch * beam, 1] -> [batch, beam]
                last_prob = paddle.reshape(last_prob, [-1, beam_size])
                # print("last_prob",last_prob)

                # [batch *beam,1]
                # batch_beam_index = paddle.reshape(batch_beam_index, [-1,1])
                # batch_beam_index = paddle.expand(beam_index, [batch_size, beam_size])
                # [i,beam_size]
                saved_path_index = paddle.expand(beam_index, [saved_path.shape[0], beam_size])
                # print("saved_path_index",saved_path_index)
                saved_path = paddle.index_sample(saved_path, saved_path_index)
                # print("saved_path_indexSamp",saved_path)
                # for j in range(len(prob_list)):
                #     prob_list[j] = paddle.index_sample(prob_list[j], batch_beam_index)

                # [i + 1, beam_size]
                saved_path = paddle.concat([saved_path, item_index], axis=0)
                # print("saved_path_concatItem",saved_path)

            # [beam, width]
        saved_path = paddle.transpose(saved_path, [1, 0])
        # print("out_tran_saved_path:", saved_path)
        # [batch, beam] -> [beam]
        final_prob = paddle.sum(last_prob, axis=0)
        # print("final_prob:", final_prob)
        return saved_path, final_prob

    def forward(self, hist_item_seq=None, hist_cat_seq=None, user_embedding=None, kd_label=None, multi_task_positive_labels=None,
                multi_task_negative_labels=None, mask=None, is_infer=False, re_infer=False):
        
        # print("-----+++++-----forward-----+++++--------")

        if self.data_format =="user_embedding":
            user_embedding = user_embedding   # shape=[2, 2]
        elif self.data_format =="amazon_book_behavior":
            # print("-----hist_item_seq", hist_item_seq)
            hist_item_tensor = paddle.to_tensor(np.array(hist_item_seq))
            hist_cat_tensor = paddle.to_tensor(np.array(hist_cat_seq))
            # print("-----hist_item_tensor", hist_item_tensor)
            hist_item_tensor = paddle.cast(hist_item_tensor, dtype='int64')
            hist_cat_tensor = paddle.cast(hist_cat_tensor, dtype='int64')
            # print("-----hist_item_tensor", hist_item_tensor)
            # print("--+++++-----forward-----+++++---hist_item_seq--tensor???",hist_item_seq)
            # paddle.static.Print(hist_item_seq)
            # print("--+++++-----forward-----+++++---hist_cat_seq",hist_cat_seq)
            # print("--+++++-----forward-----+++++---hist_cat_tensor",hist_cat_tensor)

            hist_item_emb = self.hist_item_emb_attr(hist_item_tensor)
            hist_cat_emb = self.hist_cat_emb_attr(hist_cat_tensor)
            # print("hist_item_emb-------:",hist_item_emb) 
            # paddle.static.Print(hist_item_emb)
            # print("hist_cat_emb-------:",hist_cat_emb) 

            hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
            # print("hist_seq_concat",hist_seq_concat)
            # print("hist_item_emb",hist_item_emb)
            # print("hist_cat_emb",hist_cat_emb)
            seq_shape = hist_seq_concat.shape
            seq_len=seq_shape[1]
            # gru_linear = paddle.nn.Linear(
            #     in_features=seq_len,
            #     out_features=1)

            # print("----hist_seq_concat---",hist_seq_concat.shape)
            # print("seq_shape",seq_shape)
            # print("seq_len",seq_len)
            # print("hist_seq_concat-------:",hist_seq_concat.shape) 
            batchSize = seq_shape[0]
            # print("user_seq_len-------:",user_seq_len)   # how to set gru_input??
            # gru_input=paddle.reshape(hist_seq_concat, shape = [-1, user_seq_len, self.gru_input_size])    # tensor [1, 2, 128]

            # layer_norm = paddle.nn.LayerNorm(hist_seq_concat.shape)
            # layer_norm_out = layer_norm(hist_seq_concat)

            gru_input = hist_seq_concat
            # gru_soft= paddle.nn.Softmax()
            # res.append(np.array(mask[i]).astype('int64'))
            # print("hist_seq_concat.shape", hist_seq_concat.shape)
            # print("mask.shape",mask.shape)
            # mask_hist = paddle.fluid.layers.elementwise_add(hist_seq_concat, mask)
            # hist_mask = gru_soft(hist_seq_concat)
            # print("gru_input-------:",gru_input) 
            # gru_init = paddle.randn((2,batchSize,self.user_embedding_size))
            # print("gru_init",gru_init.shape)
            # print("self.gru",self.gru)
            # gru_out, cur_status = self.gru(gru_input,gru_init)   # tensor [1, 2, 2]; tensor [1, 1, 2]
            gru_out, cur_status = self.gru(gru_input)   # tensor [1, 2, 2]; tensor [1, 1, 2]
            # print("cur_status",cur_status)
            # user_embedding_status = paddle.sum(cur_status, axis=0)
            # print("user_embedding_status", user_embedding_status)
            # print("gru_out", gru_out)
            user_embedding = paddle.sum(gru_out, axis=1)
            # print("user_embedding", user_embedding)
            
        def train_forward(user_embedding, kd_label=None, multi_task_positive_labels=None,
                          multi_task_negative_labels=None, mask=None):

            # print("----------train_forward-------------")
            # print("----------user_embedding", user_embedding)
            # print("----------pre_kd_label", kd_label)
            # print("----------multi_task_positive_labels", multi_task_positive_labels)
            # print("----------multi_task_negative_labels", multi_task_negative_labels)    #  neg_labels: 100*100 ????

            kd_label = paddle.reshape(kd_label, [-1, self.width])
            # print("----------reshape_kd_label", kd_label)

            path_emb_idx_lists = []
            for idx in range(self.width):
                cur_path_emb_idx = paddle.slice(
                    kd_label, axes=[1], starts=[idx], ends=[idx + 1])  # (batch_size * J, 1)
                # print("cur_path_emb_idx.shape", cur_path_emb_idx)
                # print("cur_path_emb_idx shape",batch_size *J," 1") 
                path_emb_idx_lists.append(cur_path_emb_idx)

                # print("path_emb_idx.shape", cur_path_emb_idx.shape)
            # Lookup table path emb
            # The main purpose of two-step table lookup is for distributed PS training
            path_emb = []
            for idx in range(self.width):
                emb = self.path_embedding(
                    path_emb_idx_lists[idx])  # (batch_size * J, 1, emb_shape)
                path_emb.append(emb)

                # print("emb_shape ", emb.shape)

            # expand user_embedding (batch_size, emb_shape) -> (batch_size * J, emb_shape)

            input_embedding = self.expand_layer(
                user_embedding, self.item_path_volume)

            # print("user_embedding", user_embedding.shape)
            # print("item_path_volume ", self.item_path_volume)
            # print("input_embedding ", input_embedding.shape)
            # print("self.expand_layer", self.expand_layer)

            # calc prob of every layer
            path_prob_list = []
            for i in range(self.width):
                cur_input_list = []
                cur_input = None
                # input: user emb + c_d_emb
                if i == 0:
                    cur_input = input_embedding
                else:
                    cur_input_list.append(input_embedding)
                    for j in range(i):
                        cur_input_list.append(paddle.reshape(
                            path_emb[j], (-1, self.user_embedding_size)))
                    cur_input = paddle.concat(cur_input_list, axis=1)

                # print("---------cur_input",cur_input)
                # print("---------mlp_layers",self.mlp_layers)
                # print("---------mlp_layers[i]",self.mlp_layers[i])

                layer_prob = F.softmax(self.mlp_layers[i](cur_input))

                # print("---------layer_prob",layer_prob)

                cur_path_prob = paddle.index_sample(
                    layer_prob, path_emb_idx_lists[i])  # (batch_size * J, 1)
                path_prob_list.append(cur_path_prob)

            path_prob = paddle.concat(
                path_prob_list, axis=1)  # (batch_size * J, D)
            # print("----path_prob", path_prob)

            multi_task_loss = None
            if self.use_multi_task_learning:
                temp = user_embedding
                # print("user_embedding",user_embedding)

                # (batch, dot_product_size)
                for i in range(len(self.multi_task_mlp_layers)):
                    temp = self.multi_task_mlp_layers[i](temp)
                # print("self.multi_task_mlp_layers",self.multi_task_mlp_layers)
                # print("temp",temp)

                # (batch, dot_product_size)
                pos_item_embedding = self.multi_task_item_embedding(multi_task_positive_labels)
                neg_item_embedding = self.multi_task_item_embedding(multi_task_negative_labels)
                # print("---------self.multi_task_item_embedding",self.multi_task_item_embedding)
                # print("---------pos_item_embedding",pos_item_embedding)
                # print("---------neg_item_embedding",neg_item_embedding)
                # print("---------self.dot_product_size",self.dot_product_size)
                # paddle.static.Print(multi_task_negative_labels, message = "multi_task_negative_labels")
                # paddle.static.Print(neg_item_embedding, message = "neg_item_embedding")

                neg_item_embedding = paddle.fluid.layers.reduce_mean(neg_item_embedding, dim=1)
                # paddle.static.Print(neg_item_embedding, message = "reduce_xneg_item_embedding")

                pos_item_embedding = paddle.reshape(pos_item_embedding, [-1, self.dot_product_size])
                neg_item_embedding = paddle.reshape(neg_item_embedding, [-1, self.dot_product_size])
                # print("---------pos_item_embedding",pos_item_embedding)
                # print("---------neg_item_embedding",neg_item_embedding)
                # paddle.static.Print(temp, message = "temp")
                # paddle.static.Print(neg_item_embedding,  message = "neg_item_embedding")

                # (batch,1)
                # print("------------temp",temp)
                pos = paddle.dot(temp, pos_item_embedding)
                # print("------------neg_item_embedding",neg_item_embedding)
                neg = paddle.dot(temp, neg_item_embedding)
                # print("---------pos",pos)
                # print("---------neg",neg)
                neg = paddle.clip(x=neg, min=-15, max=15)
                # neg = paddle.clip(x=neg, min=-200, max=200)
                pos = paddle.clip(x=pos, min=-15, max=15)


                pos = paddle.log(paddle.nn.functional.sigmoid(pos))
                neg = paddle.log(1 - paddle.nn.functional.sigmoid(neg))
                # (batch,2)
                sum = paddle.concat([pos, neg], axis=1)
                multi_task_loss = paddle.sum(sum)[0]
                multi_task_loss = multi_task_loss * -1
            # print("---------log-pos",pos)
            # print("---------log-neg",neg)
            # print("---------path_prob",path_prob)
            # print("---------multi_task_loss",multi_task_loss)

            return path_prob, multi_task_loss

        def infer_forward(user_embedding):
            height = paddle.full(
                shape=[1, 1], fill_value=self.height, dtype='int64')

            prev_index = []
            path_prob = []

            for i in range(self.width):
                if i == 0:
                    # first layer, input only use user embedding
                    # user-embedding [batch, emb_shape]
                    # [batch, K]
                    tmp_output = F.softmax(self.mlp_layers[i](user_embedding))
                    # assert beam_search_num < height
                    # [batch, B]
                    prob, index = paddle.topk(
                        tmp_output, self.beam_search_num)
                    path_prob.append(prob)

                    # expand user_embedding (batch_size, emb_shape) -> (batch_size * B, emb_shape)
                    # print("user_embedding shape ",user_embedding.shape)
                    # print("beam search num ",self.beam_search_num)
                    input_embedding = self.expand_layer(
                        user_embedding, self.beam_search_num)
                    prev_index.append(index)
                    # print("fist prev_index: {}".format(prev_index))

                else:
                    # other layer, use user embedding + path_embedding
                    # (batch_size * B, emb_size * N)
                    # cur_layer_input = paddle.concat(prev_embedding, axis=1)
                    input = input_embedding
                    for j in range(len(prev_index)):
                        # [batch,beam,emb_size]
                        emb = self.path_embedding(prev_index[j])
                        # [batch*beam,emb_size]
                        emb = paddle.reshape(emb, [-1, self.user_embedding_size])
                        input = paddle.concat([input, emb], axis=1)

                    # (batch_size * B, K)
                    # tmp_output = F.softmax(self.mlp_layers[i](cur_layer_input))
                    tmp_output = F.softmax(self.mlp_layers[i](input))
                    # (batch_size, B * K)
                    tmp_output = paddle.reshape(
                        tmp_output, (-1, self.beam_search_num * self.height))
                    # (batch_size, B)
                    prob, index = paddle.topk(
                        tmp_output, self.beam_search_num)
                    # path_prob.append(prob)

                    # prev_index of B
                    # print("index: {}".format(index))
                    prev_top_index = paddle.floor_divide(index, height)
                    # print("prev_top_index: {}".format(prev_top_index))
                    for j in range(len(prev_index)):  #
                        prev_index[j] = paddle.index_sample(prev_index[j], prev_top_index)
                        path_prob[j] = paddle.index_sample(path_prob[j], prev_top_index)
                    path_prob.append(prob)
                    cur_top_abs_index = paddle.mod(index, height)
                    prev_index.append(cur_top_abs_index)
                    # print("cur_top_abs_index: {}".format(cur_top_abs_index))

            final_prob = path_prob[0]
            for i in range(1, len(path_prob)):
                final_prob = paddle.multiply(final_prob, path_prob[i])
            for i in range(len(prev_index)):
                # [batch,beam,1]
                # print("---prev_index[i]", prev_index[i])
                prev_index[i] = paddle.reshape(prev_index[i], [-1, self.beam_search_num, 1])
                # print("---beamSearchShape--prev_index[i]", prev_index[i])


            # [batch,beam,width],[batch,beam]
            kd_path = paddle.concat(prev_index, axis=-1,name="kd_path")
            # print("kd_path", kd_path)
            # print("final_prob", final_prob)
            return kd_path, final_prob, user_embedding

        def mul_infer_forward(user_embedding):
	        # print("re_infer",re_infer)
	        # print("bs_item_list",bs_item_list)
            layer_embedding = user_embedding   
            # batch_size = len(user_embedding[0])                                                             
            # (batch, dot_product_size)
            for i in range(len(self.multi_task_mlp_layers)):
                layer_embedding = self.multi_task_mlp_layers[i](layer_embedding)
            # user_pros={userEmb: [(itemId,pro),(item,pro),...]}
            user_pros={}
            for batch_index in range(len(bs_item_list)):
                batch_layer_embedding=layer_embedding[batch_index]  # user_embedding in a batch index   
                batch_user_embedding=user_embedding[batch_index]             
                user_pros[batch_user_embedding]=[]
                
                for rec_item_index in range(1,len(bs_item_list[batch_index])):  # item_index
                    # print(bs_item_list[batch_index][rec_item_index])
                    item_tensor = paddle.to_tensor(bs_item_list[batch_index][rec_item_index])
                    bs_item_embedding = self.multi_task_item_embedding(item_tensor)
                    # bs_item_embedding = paddle.concat(bs_item_embedding, )
                    # print("batch_layer_embedding",batch_layer_embedding)
                    # print("bs_item_embedding",bs_item_embedding[0])
                    pro_item = paddle.dot(batch_layer_embedding,bs_item_embedding[0])
                    # use_item_pro;
                    # print("pro_item",pro_item)
                    # print("pro_item_numpy",pro_item.numpy()[0])
                    user_pros[batch_user_embedding].append([bs_item_list[batch_index][rec_item_index] , pro_item.numpy()[0]])
                    # print("user_pros",user_pros)
                    # print("rec_result",user_pros[batch_user_embedding])
                user_pros[batch_user_embedding].sort(reverse=True, key = lambda x: x[1])
            #     print(user_pros[batch_user_embedding])
            #     print(user_pros)
            # print(user_pros)
            return user_pros
    #    {  Tensor(shape=[2], dtype=float32, place=CPUPlace, stop_gradient=True,
    #           [0.36140999, 0.02512000]): [], 
    #       Tensor(shape=[2], dtype=float32, place=CPUPlace, stop_gradient=True,
    #           [0.12523000, 0.31525001]): [[10, 10.810221], [9, 6.3761997]]}

        if is_infer:
            return infer_forward(user_embedding)
        elif re_infer:
            # print("-----re_infer------")
            return mul_infer_forward(user_embedding)        
        else:
            return train_forward(user_embedding, kd_label, multi_task_positive_labels, multi_task_negative_labels)
