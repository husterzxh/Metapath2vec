#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import random
import numpy as np


class VocabularyGenerator(object):
    def __init__(self, meta_path_txt, window_size):
        dict_index_node, dict_node_index, dict_node_counts, dict_index_counts, \
        list_node_context, dict_index_type, dict_type_index = \
            self.parse_meta_path_txt(meta_path_txt, window_size)

        self.window_size = window_size
        self.dict_index_node = dict_index_node
        self.dict_node_index = dict_node_index
        self.dict_node_counts = dict_node_counts
        self.dict_index_counts = dict_index_counts
        self.list_node_context = list_node_context
        self.dict_index_type = dict_index_type
        self.dict_type_index = dict_type_index

        self.prepare_sampling_dist(dict_index_counts, dict_index_type, dict_type_index)
        # 将列表打乱
        random.shuffle(self.list_node_context)
        self.count = 0
        self.epoch = 1

    # 解析元路径文件
    def parse_meta_path_txt(self, meta_path_txt, window_size):
        # 统计每个节点的个数
        dict_node_counts = {}
        with open(meta_path_txt, encoding='latin-1') as f:
            for line in f:
                list_nodes = [node.strip() for node in line.strip().split(' ')]
                for item_sent in list_nodes:
                    if len(item_sent) == 0:
                        continue
                    if dict_node_counts.__contains__(item_sent):
                        dict_node_counts[item_sent] += 1
                    else:
                        dict_node_counts[item_sent] = 1
        # 索引-节点
        dict_index_node = dict((index, node) for index, node in enumerate(dict_node_counts.keys()))
        # 节点-索引
        dict_node_index = dict((node, index) for index, node in dict_index_node.items())
        # 索引-出现个数
        dict_index_counts = dict((dict_node_index[node], counts) for node, counts in dict_node_counts.items())
        # 索引-节点类型
        dict_index_type = dict((index, node[0]) for index, node in dict_index_node.items())
        # 节点类型-索引
        dict_type_index = {}
        for item_type in set(dict_index_type.values()):
            dict_type_index[item_type] = []
        for item_index, item_type in dict_index_type.items():
            dict_type_index[item_type].append(item_index)
        for item_type in dict_type_index:
            dict_type_index[item_type] = np.array(dict_type_index[item_type])

        # 围绕中心节点构造上下文
        list_node_context = []
        with open(meta_path_txt, encoding='latin-1') as f:
            for line in f:
                list_indexes = [dict_node_index[node.strip()] for node in line.split(' ') if node.strip() in dict_node_index]
                for position_center_node, index_center_node in enumerate(list_indexes):
                    start = max(0, position_center_node - window_size)
                    end = min(len(list_indexes), position_center_node + window_size + 1)
                    list_context = list_indexes[start: position_center_node] + list_indexes[position_center_node + 1: end + 1]
                    for item_index in list_context:
                        list_node_context.append((index_center_node, item_index))

        return dict_index_node, dict_node_index, dict_node_counts, dict_index_counts, list_node_context, \
               dict_index_type, dict_type_index

    def prepare_sampling_dist(self, dict_index_counts, dict_index_type, dict_type_index):
        sampling_prob = np.zeros(len(dict_index_counts))
        for i in range(len(dict_index_counts)):
            sampling_prob[i] = dict_index_counts[i]
        sampling_prob = sampling_prob ** (3.0 / 4.0)

        all_types = set(dict_index_type.values())
        dict_type_probs = {}
        for node_type in all_types:
            indicies_for_a_type = dict_type_index[node_type]
            dict_type_probs[node_type] = np.array(sampling_prob[indicies_for_a_type])
            dict_type_probs[node_type] = dict_type_probs[node_type] / np.sum(dict_type_probs[node_type])

        sampling_prob = sampling_prob / np.sum(sampling_prob)
        self.sampling_prob = sampling_prob
        self.dict_type_probs = dict_type_probs

    def get_one_batch(self):
        if self.count == len(self.list_node_context):
            self.count = 0
            self.epoch += 1
        node_context_pair = self.list_node_context[self.count]
        self.count += 1
        return node_context_pair

    def get_batch(self, batch_size):
        pairs = np.array([self.get_one_batch() for i in range(batch_size)])
        return pairs[:, 0], pairs[:, 1]

    def get_negative_samples(self, pos_index, num_negatives, care_type):
        pos_prob = self.sampling_prob[pos_index]
        if not care_type:
            negative_samples = np.random.choice(len(self.dict_index_node), size=num_negatives, replace=False,
                                                p=self.sampling_prob)
            negative_probs = self.sampling_prob[negative_samples]
        else:
            node_type = self.dict_index_type[pos_index]
            sampling_probs = self.dict_type_probs[node_type]
            sampling_candidates = self.dict_type_index[node_type]
            negative_samples_indices = np.random.choice(len(sampling_candidates), size=num_negatives, replace=False,
                                                        p=sampling_probs)

            negative_samples = sampling_candidates[negative_samples_indices]
            negative_probs = sampling_probs[negative_samples_indices]

        # print(negative_samples,pos_prob,negative_probs)
        return negative_samples, pos_prob.reshape((-1, 1)), negative_probs

    # 将列表打乱
    def shffule(self):
        random.shuffle(self.list_node_context)