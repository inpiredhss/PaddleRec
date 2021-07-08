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

__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../tools')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader, reset_auc
from utils.save_load import save_model, load_model
from operator import itemgetter, attrgetter
from paddle.io import DistributedBatchSampler, DataLoader
import paddle.fluid as fluid
import argparse
import numpy as np
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_loader1(config, place, graph_index, mode):
    if mode == "train":
        data_dir = config.get("runner.train_data_dir", None)
        batch_size = config.get('runner.train_batch_size', None)
        reader_path = config.get('runner.train_reader_path', 'reader')
    else:
        # data_dir = config.get("runner.train_data_dir", None)
        # batch_size = config.get('runner.train_batch_size', None)
        # reader_path = config.get('runner.train_reader_path', 'reader')
        data_dir = config.get("runner.test_data_dir", None)
        batch_size = config.get('runner.infer_batch_size', None)
        reader_path = config.get('runner.infer_reader_path', 'reader')
    config_abs_dir = config.get("config_abs_dir", None)
    data_dir = os.path.join(config_abs_dir, data_dir)
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    user_define_reader = config.get('runner.user_define_reader', False)
    logger.info("reader path:{}".format(reader_path))
    from importlib import import_module
    reader_class = import_module(reader_path)
    dataset = reader_class.RecDataset(file_list, config=config, graph_index = graph_index)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=True)
    return loader

def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args

def main(args):
    paddle.seed(12345)
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
    # print("---dy_model_class", dy_model_class)
    config["config_abs_dir"] = args.abs_dir    
    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    test_data_dir = config.get("runner.test_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)

    reader_path = config.get('runner.infer_reader_path', 'reader')
    item_path_volume = config.get("hyper_parameters.item_path_volume")
    batch_size = config.get("runner.train_batch_size", None)
    width = config.get("hyper_parameters.width")
    recall_num = config.get("hyper_parameters.recall_num")
    user_embedding_size = config.get(
        "hyper_parameters.user_embedding_size")
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    logger.info("**************common.configs**********")


    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    dy_model = dy_model_class.create_model(config)
    # print("dy_model")

    # to do : add optimizer function
    optimizer = dy_model_class.create_optimizer(dy_model, config)
    graph_index = dy_model_class.graph_index._graph  


    logger.info("read data")
    # print("----dy_model_class.graph_index", dy_model_class.graph_index)
    test_dataloader = create_data_loader1(config=config, place=place, graph_index=dy_model_class.graph_index,mode="test")
    # print("----test_dataloader",test_dataloader)
    # train_dataloader = create_data_loader1(config=config, place=place, graph_index=dy_model_class.graph_index)

#    epoch_begin = time.time()
 #   interval_begin = time.time()
  #  metric_list, metric_list_name = dy_model_class.create_metrics()

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, dy_model)
        dy_model.eval()

        metric_list, metric_list_name = dy_model_class.create_metrics()
        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        # print("-------before: data")
        for batch_id, batch in enumerate(test_dataloader()):

            # print('---batch',batch)
            batch_user = batch[:2]
            label_items = batch[-1]
            # print("--batch_user--", batch_user)
            # print("---label_items--", label_items)
            item_path_prob, presision, recall, F1 = dy_model_class.infer_forward(
                dy_model, batch, config)

            if batch_id % print_interval == 0:
                logger.info("epoch: {}, batch_id: {}, presision {}, recall {}, F1 {}".format(
                    epoch_id, batch_id, presision, recall, F1) +
                            " speed: {:.2f} ins/s".format(
                                print_interval * batch_size / (time.time(
                                ) - interval_begin)))
                interval_begin = time.time()

if __name__ == '__main__':
    args = parse_args()
    main(args)
