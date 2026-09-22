import random
from utils.DataLoader import Data
import matplotlib.pyplot as plt
import collections
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from collections import OrderedDict
import numpy as np
from scipy.stats import trim_mean
from tqdm import tqdm
import pickle


class NeighborScorer(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, fuse_embeds):
        # node_embed: [B, D]; nbr_embeds: [B, N, D]; time_feats: [B, N, T]
        logits = self.mlp(fuse_embeds).squeeze(-1)  # -> [B, N]
        return logits


class TimeAwareSampler(nn.Module):
    def __init__(self, embed_dim, time_dim, hidden_dim, num_neighbors, tau_init=1.0):
        super().__init__()
        self.scorer = NeighborScorer(2 * embed_dim + time_dim, hidden_dim)
        self.num_neighbors = num_neighbors
        self.tau = nn.Parameter(torch.tensor(tau_init))

    def concrete_sample(self, log_alpha, beta=1.0, training=False):
        if training:
            bias = 0.1
            noise = torch.empty_like(log_alpha).uniform_(bias, 1 - bias)
            gate_inputs = torch.log(noise) - torch.log(1 - noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            return torch.sigmoid(gate_inputs)
        else:
            return torch.sigmoid(log_alpha)

    def forward(self, node_embeds):
        """
        node_embeds: [B, D]
        """
        log_alpha = self.scorer(node_embeds).squeeze(-1)
        neighbor_probs = self.concrete_sample(log_alpha, training=self.training)
        return neighbor_probs


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


def create_optimizer_new(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0,
                         idx: int = 0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if idx == 0 or idx == 1:
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=model[idx].parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=model[idx].parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(params=model[idx].parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Wrong value for optimizer {optimizer_name}!")
    else:
        print(model)
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


def plot_number_frequency(arr_list, name):
    combined_arr = np.concatenate(arr_list)
    unique_numbers, counts = np.unique(combined_arr, return_counts=True)
    plt.bar(unique_numbers, counts)
    plt.xlabel('number')
    plt.ylabel('frequency')
    plt.title(name)
    plt.show()
    plt.close()
    return unique_numbers, counts


def time_proximity_removal(edges, tau):
    """
    时间邻近去重
    :param edges: 边列表，每个元素为 (节点1, 节点2, 时间戳)
    :param tau: 时间窗口阈值
    :return: 去重后的边列表
    """
    edges.sort(key=lambda x: (x[0], x[1], x[2]))
    result = []
    prev_edge = None
    for edge in edges:
        if prev_edge is None or edge[0] != prev_edge[0] or edge[1] != prev_edge[1] or edge[2] - prev_edge[2] > tau:
            result.append(edge)
            prev_edge = edge
    return result


# class LRUCache:
#     def __init__(self, capacity: int):
#         self.capacity = capacity
#         self.cache = OrderedDict()
#
#     def get(self, key):
#         if key not in self.cache:
#             return None
#         self.cache.move_to_end(key)
#         return self.cache[key]
#
#     def put(self, key, value):
#         if key in self.cache:
#             del self.cache[key]
#         elif len(self.cache) >= self.capacity:
#             self.cache.popitem(last=False)
#         self.cache[key] = value

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.total_accesses = 0
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key):
        self.total_accesses += 1  # 记录一次访问
        if key not in self.cache:
            self.miss_count += 1  # 未命中
            return None
        # 命中情况
        self.hit_count += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.total_accesses += 1  # 记录一次访问
        if key in self.cache:
            # 键已存在，属于命中
            self.hit_count += 1
            del self.cache[key]
        else:
            # 新键，属于未命中
            self.miss_count += 1
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def get_hit_rate(self):
        """计算并返回命中率"""
        if self.total_accesses == 0:
            return 0.0
        return self.hit_count / self.total_accesses

    def get_miss_rate(self):
        """计算并返回未命中率"""
        if self.total_accesses == 0:
            return 0.0
        return self.miss_count / self.total_accesses

    def reset_stats(self):
        """重置所有统计数据"""
        self.total_accesses = 0
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self):
        """返回所有统计信息的字典"""
        return {
            'total_accesses': self.total_accesses,
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': self.get_hit_rate(),
            'miss_rate': self.get_miss_rate()
        }


def remove_redundant_timestamps(timestamps, nodes, window_size):
    combined = sorted(zip(timestamps, nodes), key=lambda x: x[0])
    result = []
    drop_result = []
    for t, node in combined:
        window_start = t - window_size
        has_duplicate = False
        i = len(result) - 1
        while i >= 0 and result[i][0] >= window_start:
            if result[i][1] == node:
                has_duplicate = True
                break
            i -= 1
        if not has_duplicate:
            result.append((t, node))
        else:
            drop_result.append((t, node))
    filtered_timestamps = [t for t, node in result]
    filtered_nodes = [node for t, node in result]
    drop_filtered_timestamps = [t for t, node in drop_result]
    drop_filtered_nodes = [node for t, node in drop_result]
    return filtered_timestamps, filtered_nodes, drop_filtered_timestamps, drop_filtered_nodes


def get_window_size(timestamps, p=0.9, method="mean"):
    # intervals = np.diff(timestamps)
    intervals = timestamps
    mean_interval = np.mean(intervals)
    if method == "mean":
        return mean_interval
    elif method == "trim":
        value = trim_mean(intervals, proportiontocut=0.2)
        return value
    elif method == "iqr":
        Q1 = np.percentile(intervals, 25)
        Q3 = np.percentile(intervals, 50)
        filtered_data = [x for x in intervals if x <= Q3]
        value = np.mean(filtered_data)
        return value
    elif method == "poisson":
        lambda_ = 1 / mean_interval
        w = -np.log(1 - p) / lambda_
        return w


def sample_window_size(node_ids_list, p=0.9, method="mean"):
    n = len(node_ids_list)
    result = []
    count = 0
    while count < 5:
        i = random.sample(range(1, n + 1), 1)[0]
        if len(node_ids_list[i]) > 1:
            tmp = list(np.diff(node_ids_list[i]))
            result = result + tmp
            count += 1
    value = get_window_size(result, p, method)
    return value


class NeighborSampler:
    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0,
                 seed: int = None, temporal_pagerank: list = None, sample_args=None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []
        self.max_neighbor_number = None
        self.sample_args = sample_args
        self.bypass_time=0

        if self.sample_neighbor_strategy == "our" and sample_args.edge_sampling == 1:
            self.edge_sampler = TimeAwareSampler(sample_args.node_dim, sample_args.time_dim, 10,
                                                 sample_args.num_neighbors, tau_init=1.0)
        else:
            self.edge_sampler = None

        self.bypass=sample_args.bypass

        if sample_args.cache == 1:
            self.cache = True
            max_size = 1000
            self.lru_cache = LRUCache(capacity=max_size)
            self.cache_num = sample_args.num_neighbors
        else:
            self.cache = False

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in tqdm(enumerate(adj_list)):
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.sample_neighbor_strategy == 'all_sample':
            n = [len(self.nodes_neighbor_ids[i]) for i in range(len(self.nodes_neighbor_ids))]
            self.max_neighbor_number = max(n)
            p = np.percentile(n, 80)
            self.max_neighbor_number = int(p)

        if self.sample_neighbor_strategy == 'our':
            previous_edge = np.sum([len(self.nodes_neighbor_ids[i]) for i in range(len(self.nodes_neighbor_ids))])
            window_size = sample_window_size(self.nodes_neighbor_times, 0.9, "iqr")
            for node_idx, per_node_neighbors in enumerate(adj_list):
                neighbour_size = len(self.nodes_neighbor_times[node_idx])
                if neighbour_size == 0:
                    continue
                node_sample_size = math.ceil((neighbour_size * self.sample_args.presampling_total_rate))
                # Pre-sampling phase
                # window_size = 100
                filtered_timestamps, filtered_nodes, drop_filtered_timestamps, drop_filtered_nodes = remove_redundant_timestamps(
                    self.nodes_neighbor_times[node_idx], self.nodes_neighbor_ids[node_idx], window_size)

                if filtered_nodes is not np.nan and len(self.nodes_neighbor_ids[node_idx]) > 0:
                    self.nodes_neighbor_times[node_idx] = np.array(filtered_timestamps)
                    self.nodes_neighbor_ids[node_idx] = np.array(filtered_nodes)
                if temporal_pagerank is not None:
                    if filtered_nodes is not np.nan and len(self.nodes_neighbor_ids[node_idx]) > 0 and len(
                            filtered_timestamps) - node_sample_size > 0:
                        weights = temporal_pagerank[self.nodes_neighbor_ids[node_idx]]
                        # node_sample_size = int(len(filtered_nodes) * self.sample_args.presampling_rate)
                        normalized_weights = np.array(weights) / np.sum(weights)
                        sampled_indices = np.random.choice(len(filtered_nodes), size=max(node_sample_size, 1),
                                                           replace=False, p=normalized_weights)
                        sorted_indices = np.array(sorted(sampled_indices))
                        filtered_timestamps = [filtered_timestamps[i] for i in sorted_indices]
                        filtered_nodes = [filtered_nodes[i] for i in sorted_indices]
                        self.nodes_neighbor_times[node_idx] = np.array(filtered_timestamps)
                        self.nodes_neighbor_ids[node_idx] = np.array(filtered_nodes)
            current_edge = np.sum([len(self.nodes_neighbor_ids[i]) for i in range(len(self.nodes_neighbor_ids))])
            print("Reduce ratio:", current_edge / previous_edge)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

        if self.sample_args.save_pkl == 1:
            with open(f"nodes_neighbor_ids_{self.sample_args.dataset_name}.pkl", "wb") as f:
                pickle.dump(self.nodes_neighbor_ids, f)
            with open(f"nodes_edge_ids_{self.sample_args.dataset_name}.pkl", "wb") as f:
                pickle.dump(self.nodes_edge_ids, f)
            with open(f"nodes_neighbor_times_{self.sample_args.dataset_name}.pkl", "wb") as f:
                pickle.dump(self.nodes_neighbor_times, f)


    def custom_hash(self, node_id, interact_time):
        """生成缓存键的哈希函数"""
        return hash((node_id, interact_time))

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[
                                                                                               node_id][:i], \
                self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[
                                                                                               node_id][:i], None

    def get_result(self, node_id: int, node_interact_time: time, num_neighbors: int = 20):
        # nodes_neighbor_ids = np.zeros((num_neighbors)).astype(np.longlong)
        # nodes_edge_ids = np.zeros((bsz, num_neighbors)).astype(np.longlong)
        # nodes_neighbor_times = np.zeros((bsz, num_neighbors)).astype(np.float32)

        node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
            self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time,
                                       return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')
        nodes_neighbor_ids = np.zeros(num_neighbors).astype(np.longlong)
        nodes_edge_ids = np.zeros(num_neighbors).astype(np.longlong)
        nodes_neighbor_times = np.zeros(num_neighbors).astype(np.float32)
        if len(node_neighbor_ids) > 0:
            if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                if node_neighbor_sampled_probabilities is not None:
                    # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                    # torch.softmax() function can tackle this case
                    node_neighbor_sampled_probabilities = torch.softmax(
                        torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                if self.seed is None:
                    sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                       p=node_neighbor_sampled_probabilities)
                else:
                    sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                               p=node_neighbor_sampled_probabilities)

                nodes_neighbor_ids = node_neighbor_ids[sampled_indices]
                nodes_edge_ids = node_edge_ids[sampled_indices]
                nodes_neighbor_times = node_neighbor_times[sampled_indices]

                # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                sorted_position = nodes_neighbor_times.argsort()
                nodes_neighbor_ids = nodes_neighbor_ids[sorted_position]
                nodes_edge_ids = nodes_edge_ids[sorted_position]
                nodes_neighbor_times = nodes_neighbor_times[sorted_position]
            elif self.sample_neighbor_strategy in ['recent', 'all_sample']:
                # elif self.sample_neighbor_strategy in ['recent']:
                # Take most recent interactions with number num_neighbors
                node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                node_edge_ids = node_edge_ids[-num_neighbors:]
                node_neighbor_times = node_neighbor_times[-num_neighbors:]
                # print("SS", time.time() - t1)

                # put the neighbors' information at the back positions
                nodes_neighbor_ids[num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                nodes_edge_ids[num_neighbors - len(node_edge_ids):] = node_edge_ids
                nodes_neighbor_times[num_neighbors - len(node_neighbor_times):] = node_neighbor_times
            elif self.sample_neighbor_strategy in ['our']:
                t1 = time.time()
                node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                node_edge_ids = node_edge_ids[-num_neighbors:]
                node_neighbor_times = node_neighbor_times[-num_neighbors:]

                # put the neighbors' information at the back positions
                nodes_neighbor_ids[num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                nodes_edge_ids[num_neighbors - len(node_edge_ids):] = node_edge_ids
                nodes_neighbor_times[num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                t2 = time.time()
            else:
                raise ValueError(
                    f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        if self.cache and self.cache_num == num_neighbors:
            nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
            nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
            nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

            for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
                cache_key = (node_id)
                if random.random() < self.bypass:
                    self.bypass_time+=1
                    cache_nodes_neighbor_ids, cache_nodes_edge_ids, cache_nodes_neighbor_times = self.get_result(
                        node_id, node_interact_time, num_neighbors)
                else:
                    # 尝试从cache获取
                    cached_result = self.lru_cache.get(cache_key)
                    if cached_result is not None:
                        cache_nodes_neighbor_ids, cache_nodes_edge_ids, cache_nodes_neighbor_times = cached_result
                    else:
                        # cache中没有，获取结果
                        cache_nodes_neighbor_ids, cache_nodes_edge_ids, cache_nodes_neighbor_times = self.get_result(
                            node_id, node_interact_time, num_neighbors)
                    result = (cache_nodes_neighbor_ids, cache_nodes_edge_ids, cache_nodes_neighbor_times)
                    self.lru_cache.put(cache_key, result)
                nodes_neighbor_ids[idx, :] = cache_nodes_neighbor_ids
                nodes_edge_ids[idx, :] = cache_nodes_edge_ids
                nodes_neighbor_times[idx, :] = cache_nodes_neighbor_times
        else:
            nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
            nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
            nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

            # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
            # fill_space = 0
            # full_space = 0
            for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
                # find neighbors that interacted with node_id before time node_interact_time
                node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                    self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time,
                                               return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')
                # full_space += num_neighbors
                # fill_space += min(len(node_neighbor_ids), num_neighbors)
                if len(node_neighbor_ids) > 0:
                    if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                        # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                        # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                        # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                        if node_neighbor_sampled_probabilities is not None:
                            # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                            # torch.softmax() function can tackle this case
                            node_neighbor_sampled_probabilities = torch.softmax(
                                torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                        if self.seed is None:
                            sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                               p=node_neighbor_sampled_probabilities)
                        else:
                            sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                                       p=node_neighbor_sampled_probabilities)

                        nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                        nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                        nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                        # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                        sorted_position = nodes_neighbor_times[idx, :].argsort()
                        nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                        nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                        nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                    elif self.sample_neighbor_strategy in ['recent', 'all_sample']:
                        t1 = time.time()
                        # elif self.sample_neighbor_strategy in ['recent']:
                        # Take most recent interactions with number num_neighbors
                        node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                        node_edge_ids = node_edge_ids[-num_neighbors:]
                        node_neighbor_times = node_neighbor_times[-num_neighbors:]
                        # print("SS", time.time() - t1)

                        # put the neighbors' information at the back positions
                        nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                        nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                        nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                        t2 = time.time()
                        # print("FF", t2 - t1)
                        # print("GG", len(node_neighbor_ids))
                    elif self.sample_neighbor_strategy in ['our']:
                        t1 = time.time()
                        node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                        node_edge_ids = node_edge_ids[-num_neighbors:]
                        node_neighbor_times = node_neighbor_times[-num_neighbors:]
                        # print("SS", time.time() - t1)

                        # put the neighbors' information at the back positions
                        nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                        nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                        nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                        t2 = time.time()
                        # print("FF", t2 - t1)
                        # print("GG", len(node_neighbor_ids))
                    else:
                        raise ValueError(
                            f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')
        # print("Coverrate", fill_space / full_space)
        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
                node_ids=nodes_neighbor_ids_list[-1].flatten(),
                node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


# def compute_time_redundancy(node, recent_steps=10):
#     """计算节点在最近recent_steps内的更新频率（频率越低，时间冗余越高）"""
#     updates = [t for t in self.node_features[node] if t >= (max(self.timesteps) - recent_steps)]
#     return 1 - (len(updates) / recent_steps)  # 范围 [0,1]
def get_pruned_neighbor_sampler(data: Data, edge_scores_dict: dict, pruning_ratio: float = 0.5,
                                sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0,
                                seed: int = None):
    """
    get masked neighbor sampler
    :param data: Data
    :param edge_scores_dict: dict, edge scores, where key is edge id and value is predicted score
    :param pruning_ratio: float, ratio of edges to prune
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    count = 0
    score_list = list(edge_scores_dict.values())
    c = int(len(score_list) * pruning_ratio)
    score_list.sort()
    threshold = score_list[c]

    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids,
                                                                     data.edge_ids, data.node_interact_times):
        temp_score = edge_scores_dict[edge_id]
        if temp_score > threshold:
            count += 1
            adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
            adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    print('Retain %f of edge, total %d edge' % (count / len(edge_scores_dict), len(edge_scores_dict)))

    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy,
                           time_scaling_factor=time_scaling_factor, seed=seed)



def compute_structure_redundancy(node, sampled_nodes, adj_list):
    """计算节点邻居与已采样节点的Jaccard相似度（相似度越高，结构冗余越高）"""
    neighbors = adj_list[node]
    if len(neighbors) == 0:
        return 0.0
    intersection = len(set(neighbors) & set(sampled_nodes))
    return intersection / len(neighbors)


def compute_feature_entropy(feature_vec):
    prob = np.histogram(feature_vec, bins=10, density=True)[0]
    prob = prob[prob > 0]  # 过滤零概率bin
    return -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0.0


def get_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0,
                         seed: int = None, temporal_pagerank: list = None, method: str = None, args=None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids,
                                                                     data.edge_ids, data.node_interact_times):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy,
                           time_scaling_factor=time_scaling_factor, seed=seed, temporal_pagerank=temporal_pagerank,
                           sample_args=args)


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None,
                 last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set(
                (src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in
                self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time,
                                                                               self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in
                   zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size,
                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size,
                                                                                 batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[
            random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray,
                                           batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in
                          zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size,
                                                       replace=len(possible_random_edges) < size)
        return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
            np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                        end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                           end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size,
                                                                           replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                        end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                           end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size,
                                                                          replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)
