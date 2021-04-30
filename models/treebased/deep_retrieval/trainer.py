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
import os
import paddle.nn as nn
import time
import logging
import sys
import importlib
import numpy as np
from operator import itemgetter, attrgetter
__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
print(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../tools')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from utils.save_load import load_model, save_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    item_to_user_emb = {};

    def clear_item_to_user_emb():
        item_to_user_emb = {}

    def add_item_to_user_emb(item_id,user_emb):
        if not item_id in item_to_user_emb:
            item_to_user_emb[item_id] = []
        item_to_user_emb[item_id].append(user_emb)
        print("add ",item_id," ---> ",user_emb)

    paddle.seed(12345)
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
    config["config_abs_dir"] = args.abs_dir
    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    train_data_dir = config.get("runner.train_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    em_execution_interval = config.get("hyper_parameters.em_execution_interval", 4)
    pernalize_path_to_item_count = config.get("hyper_parameters.em_execution_interval")
    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
            format(use_gpu, train_data_dir, epochs, print_interval,
                   model_save_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)

    if model_init_path is not None:
        load_model(model_init_path, dy_model)

    # to do : add optimizer function
    optimizer = dy_model_class.create_optimizer(dy_model, config)

    logger.info("read data")
    train_dataloader = create_data_loader(config=config, place=place)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode

        dy_model.train()
        metric_list, metric_list_name = dy_model_class.create_metrics()
        # auc_metric = paddle.metric.Auc("ROC")
        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        record_item_to_user_emb = True
        if (epoch_id + 1) % em_execution_interval == 0:
            record_item_to_user_emb = True
            clear_item_to_user_emb()
        for batch_id, batch in enumerate(train_dataloader()):
            if record_item_to_user_emb:
                input_emb = batch[0].numpy()
                item_set = batch[1].numpy()
                for user_emb,items in zip(input_emb,item_set):
                    add_item_to_user_emb(items[0],user_emb)

            train_reader_cost += time.time() - reader_start
            optimizer.clear_grad()
            train_start = time.time()
            batch_size = len(batch[0])

            loss, metric_list, tensor_print_dict = dy_model_class.train_forward(
                dy_model, metric_list, batch, config)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (
                            metric_list_name[metric_id] +
                            ":{:.6f}, ".format(metric_list[metric_id].accumulate())
                    )
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += (
                                "{}:".format(var_name) + str(var.numpy()) + ",")
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(
                        epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(train_reader_cost / print_interval, (
                            train_reader_cost + train_run_cost) / print_interval,
                           total_samples / print_interval, total_samples / (
                                   train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                    metric_list_name[metric_id] +
                    ": {:.6f},".format(metric_list[metric_id].accumulate()))

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    "epoch time: {:.2f} s".format(time.time() - epoch_begin))

        save_model(
            dy_model, optimizer, model_save_path, epoch_id, prefix='rec')

        if record_item_to_user_emb:
            beam_size = dy_model.beam_search_num * 4
            dy_model_class.graph_index.reset_graph_mapping()
            item_to_path_map = {}
            for key in item_to_user_emb:
                user_emb = item_to_user_emb[key]
                user_emb = paddle.to_tensor(np.array(user_emb).astype('float32'))
                path,pro = dy_model.generate_candidate_path_for_item(user_emb,beam_size)
                list = []
                path = np.array(path).astype("int64")
                pro = np.array(pro).astype("float")
                if not pernalize_path_to_item_count:
                    for single_path,single_pro in zip(path,pro):
                        list.append((single_pro,single_path))
                    sorted(list,key=itemgetter(0), reverse=True)
                    topk = []
                    for i in range(dy_model_class.item_path_volume -1):
                        print(i)
                        print(list)
                        print(list[i][1])
                        topk.append(dy_model_class.graph_index.kd_represent_to_path_id(list[i][1]))

                    topk = np.array(topk).astype("int64")
                    dy_model_class.graph_index._graph.add_item(key, topk)
                else:
                    for single_path,single_pro in zip(path,pro):
                        list.append(str(dy_model_class.graph_index.kd_represent_to_path_id(single_path)) + ":" + str(single_pro))
                    item_to_path_map[key] = list
            if pernalize_path_to_item_count:
                dy_model_class.graph_index.update_Jpath_of_item(item_to_path_map)

            dict = dy_model_class.graph_index._graph.get_item_path_dict()
            print("display new path mapping")
            for key in dict:
                print(key,"------>",dict[key])
            dy_model_class.save_item_path(model_save_path, epoch_id)
            #dy_model.save_item_path(model_save_path, epoch_id):





if __name__ == '__main__':
    args = parse_args()
    main(args)
