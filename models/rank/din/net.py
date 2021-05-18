# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.nn import Conv1D
import paddle
import paddle.nn as nn
import math
import paddle.fluid as fluid
import numpy as np
import paddle.nn.functional as F


class DINLayer(nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count):
        super(DINLayer, self).__init__()

        # self.item_emb_attr = paddle.ParamAttr(name="item_emb", initializer = fluid.initializer.Constant(value=0.0))
        # self.cat_emb_attr = paddle.ParamAttr(name="cat_emb", initializer = fluid.initializer.Constant(value=0.0))


        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.act = act
        self.is_sparse = is_sparse
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count

        self.hist_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.target_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.target_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.item_b_attr = paddle.nn.Embedding(
            self.item_count,
            1,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.attention_layer = []
        sizes = [(self.item_emb_size + self.cat_emb_size) * 4
                 ] + [8] + [4] + [1]
        acts = ["sigmoid" for _ in range(len(sizes) - 2)] + [None]

        for i in range(len(sizes) - 1):
#            flat_layer = paddle.nn.Flatten(start_axis=1, stop_axis=2)
#            self.add_sublayer('flat_%d' % i, flat_layer)
#            self.attention_layer.append(flat_layer)
            print("out_features",sizes[i + 1])
            linear = paddle.nn.Linear(
                in_features=sizes[i], #sizes[i],
                out_features=sizes[i + 1],
                # weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal()),
                # weight_attr = paddle.framework.ParamAttr(
                #     initializer=paddle.nn.initializer.XavierUniform()),
                weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
                bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0)))
            self.add_sublayer('linear_%d' % i, linear)
            self.attention_layer.append(linear)
            if acts[i] == 'sigmoid':
                # act = paddle.nn.ReLU()
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.attention_layer.append(act)
                
        self.con_layer = []
        self.firInDim = self.item_emb_size + self.cat_emb_size
        self.firOutDim = self.item_emb_size + self.cat_emb_size
        # num_flatten_dims=1

        linearCon = paddle.nn.Linear(
            in_features=self.firInDim,
            out_features=self.firOutDim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(self.firInDim))), )
        self.add_sublayer('linearCon', linearCon)
        self.con_layer.append(linearCon)

        conDim = self.item_emb_size + self.cat_emb_size + self.item_emb_size + self.cat_emb_size

        conSizes = [conDim] + [80] + [40] + [1]
        conActs = ["sigmoid" for _ in range(len(conSizes) - 2)] + [None]

        for i in range(len(conSizes) - 1):
            linear = paddle.nn.Linear(
                in_features=conSizes[i],
                out_features=conSizes[i + 1],
                weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
                bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0)))
            
                # weight_attr=paddle.ParamAttr(
                #     initializer=paddle.nn.initializer.Normal(
                #         std=1.0 / math.sqrt(conSizes[i]))), )
            self.add_sublayer('linear_%d' % i, linear)
            self.con_layer.append(linear)
            if conActs[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.con_layer.append(act)

    def forward(self, hist_item_seq, hist_cat_seq, target_item, target_cat,
                label, mask, target_item_seq, target_cat_seq):
        # print("-------hist_item_seq, hist_cat_seq, target_item, target_cat,label, mask, target_item_seq, target_cat_seq-----------")
        # print(hist_item_seq, hist_cat_seq, target_item, target_cat,label, mask, target_item_seq, target_cat_seq)
        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)
        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)
        target_item_emb = self.target_item_emb_attr(target_item)
        target_cat_emb = self.target_cat_emb_attr(target_cat)
        target_item_seq_emb = self.target_item_seq_emb_attr(target_item_seq)
        target_cat_seq_emb = self.target_cat_seq_emb_attr(target_cat_seq)
        item_b = self.item_b_attr(target_item)
        # print("-------hist_item_emb-----------")
        # print(hist_item_emb)  
        # print("-------hist_cat_emb-----------")
        # print(hist_cat_emb)           
        # print("-------target_item_emb-----------")
        # print(target_item_emb)      
        # print("-------target_cat_emb-----------")
        # print(target_cat_emb)      
        # print("-------target_item_seq_emb-----------")
        # print(target_item_seq_emb)   
        # print("-------target_cat_seq_emb-----------")
        # print(target_cat_seq_emb)      
        # print("-------item_b-----------")
        # print(item_b)   
        #                           
        hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
        # print("-------hist_seq_concat-----------")
        # print(hist_seq_concat)   

        target_seq_concat = paddle.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        # print("-------target_seq_concat-----------")
        # print(target_seq_concat) 

        target_concat = paddle.concat(
            [target_item_emb, target_cat_emb], axis=1)

        concat = paddle.concat(
            [
                hist_seq_concat, target_seq_concat,
                hist_seq_concat - target_seq_concat,
                hist_seq_concat * target_seq_concat
            ],
            axis=2)
        # print("-------concat_before_attention_shape-------", concat)
        for attlayer in self.attention_layer:
            concat = attlayer(concat)
            # print("-------concat_in_attention_shape-------", concat)
        # print("-------concat_after_attention_shape-------", concat)

            

        atten_fc3 = concat + mask
        # print("-------atten_weight_mask-------", atten_fc3)
        atten_fc3 = paddle.transpose(atten_fc3, perm=[0, 2, 1])
        # print("-------atten_weight_transpose-------", atten_fc3)
        atten_fc3 = paddle.scale(atten_fc3, scale=self.firInDim**-0.5)
        # print("-------atten_weight-------", atten_fc3)
        weight = paddle.nn.functional.softmax(atten_fc3)
        # print("-------atten_weight_softmax-------", weight)
        # print("-------hist_seq_concat-------", hist_seq_concat)

        #[b, 1, row]
        output = paddle.matmul(
            weight,
            hist_seq_concat)  # X's shape: [2, 1, 512],Y's shape: [256, 80]
        #[b,row]
        # print("-------out_matmul-------", output)
        output = paddle.reshape(output, shape=[0, self.firInDim])
        print("-------out_reshape-------", output)

        for firLayer in self.con_layer[:1]:
            concat = firLayer(output)
            print("-------con_layer-------", concat)

        embedding_concat = paddle.concat([concat, target_concat], axis=1)

        for colayer in self.con_layer[1:]:
            embedding_concat = colayer(embedding_concat)
            print("-------fc-------", embedding_concat)

        logit = embedding_concat + item_b
        print("-------logit-predict-------", logit)
        print("-------label-------", label)
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit,label)
        print("-------loss-------", loss)
        avg_loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit,label, reduction='mean')
        print("-------avg_loss-------", avg_loss)

        predict = F.sigmoid(logit)
        return avg_loss, predict
       # predict = F.sigmoid(logit)
        # print("-------sigmoid-predict-------", predict)
        #return predict


class StaticDINLayer(nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count):
        super(StaticDINLayer, self).__init__()

        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.act = act
        self.is_sparse = is_sparse
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count

        self.item_emb_attr = paddle.ParamAttr(name="item_emb")
        self.cat_emb_attr = paddle.ParamAttr(name="cat_emb")

        self.hist_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            name=self.item_emb_attr)
        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            name=self.cat_emb_attr)
        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            name=self.item_emb_attr)
        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            name=self.cat_emb_attr)
        self.target_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            name=self.item_emb_attr)

        self.target_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            name=self.cat_emb_attr)

        self.item_b_attr = paddle.nn.Embedding(
            self.item_count,
            1,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.attention_layer = []
        sizes = [(self.item_emb_size + self.cat_emb_size) * 4
                 ] + [80] + [40] + [1]
        acts = ["sigmoid" for _ in range(len(sizes) - 2)] + [None]

        for i in range(len(sizes) - 1):
            flat_layer = paddle.nn.Flatten(start_axis=0, stop_axis=sizes[i]/2)
            self.add_sublayer('flat_%d' % i, flat_layer)
            self.attention_layer.append(flat_layer)
            linear = paddle.nn.Linear(
                in_features=2, #sizes[i],
                out_features=sizes[i + 1],)
                # weight_attr=paddle.ParamAttr(
                #     initializer=paddle.nn.initializer.Normal(
                #         std=1.0 / math.sqrt(sizes[i]))), )
            self.add_sublayer('linear_%d' % i, linear)
            self.attention_layer.append(linear)
            if acts[i] == 'sigmoid':
                # act = paddle.nn.ReLU()
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.attention_layer.append(act)

        self.con_layer = []

        self.firInDim = self.item_emb_size + self.cat_emb_size
        self.firOutDim = self.item_emb_size + self.cat_emb_size
        # num_flatten_dims=1

        linearCon = paddle.nn.Linear(
            in_features=self.firInDim,
            out_features=self.firOutDim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(self.firInDim))), )
        self.add_sublayer('linearCon', linearCon)
        self.con_layer.append(linearCon)

        conDim = self.item_emb_size + self.cat_emb_size + self.item_emb_size + self.cat_emb_size

        conSizes = [conDim] + [80] + [40] + [1]
        conActs = ["relu" for _ in range(len(conSizes) - 2)] + [None]

        for i in range(len(conSizes) - 1):
            linear = paddle.nn.Linear(
                in_features=conSizes[i],
                out_features=conSizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(conSizes[i]))), )
            self.add_sublayer('linear_%d' % i, linear)
            self.con_layer.append(linear)
            if conActs[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self.con_layer.append(act)

    def forward(self, hist_item_seq, hist_cat_seq, target_item, target_cat,
                label, mask, target_item_seq, target_cat_seq):
        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)
        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)
        target_item_emb = self.target_item_emb_attr(target_item)
        target_cat_emb = self.target_cat_emb_attr(target_cat)
        target_item_seq_emb = self.target_item_seq_emb_attr(target_item_seq)
        target_cat_seq_emb = self.target_cat_seq_emb_attr(target_cat_seq)
        item_b = self.item_b_attr(target_item)

        hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
        target_seq_concat = paddle.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = paddle.concat(
            [target_item_emb, target_cat_emb], axis=1)

        # concat = paddle.concat(
        #     [hist_seq_concat, target_seq_concat, hist_seq_concat - target_seq_concat,
        #      hist_seq_concat * target_seq_concat],
        #     axis=2)

        concat = paddle.concat(
            [
                hist_seq_concat, target_seq_concat,
                paddle.elementwise_sub(hist_seq_concat, target_seq_concat),
                paddle.elementwise_mul(hist_seq_concat, target_seq_concat)
            ],
            axis=2)

        for attlayer in self.attention_layer:
            concat = attlayer(concat)

        # atten_fc3 = concat + mask

        # paddle.static.Print(mask, first_n=- 1, message=None, summarize=20, print_tensor_name=True,
        #                     print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')
        # paddle.static.Print(concat, first_n=- 1, message=None, summarize=20, print_tensor_name=True,
        #                     print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')

        atten_fc3 = paddle.elementwise_add(concat, mask)  #concat + mask
        atten_fc3 = paddle.transpose(atten_fc3, perm=[0, 2, 1])
        atten_fc3 = paddle.scale(atten_fc3, scale=self.firInDim**-0.5)
        weight = paddle.nn.functional.softmax(atten_fc3)

        # [b, 1, row]

        output = paddle.matmul(
            weight,
            hist_seq_concat)  # X's shape: [2, 1, 512],Y's shape: [256, 80]
        # [b,row]
        output = paddle.reshape(output, shape=[0, self.firInDim])

        for firLayer in self.con_layer[:1]:
            concat = firLayer(output)

        embedding_concat = paddle.concat([concat, target_concat], axis=1)

        for colayer in self.con_layer[1:]:
            embedding_concat = colayer(embedding_concat)

        logit = embedding_concat + item_b

        paddle.static.Print(
            logit,
            first_n=-1,
            message=None,
            summarize=20,
            print_tensor_name=True,
            print_tensor_type=True,
            print_tensor_shape=True,
            print_tensor_lod=True,
            print_phase='both')

        return logit
