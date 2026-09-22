from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import os
import tqdm


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                 edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float, data_root: str = "."):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv(os.path.join(data_root, 'processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)))
    edge_raw_features = np.load(os.path.join(data_root, 'processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)))
    node_raw_features = np.load(os.path.join(data_root, 'processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                     edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(
        set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    fucking_mask = np.array([(src_node_id in new_test_node_set or dst_node_id in new_test_node_set)
                             for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask],
                    labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],
                     labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask],
                              dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions,
                                                                                 full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                     edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask],
                    labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],
                     labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data


def get_node_classification_data_withtarget(dataset_name: str, val_ratio: float, test_ratio: float,
                                            target_ratio: float):
    """
    generate data for node classification task on whole test set
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
    if os.path.exists(save_file_path1) is False:
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        sampled_indices = np.random.permutation(len(train_indices))[:int(len(train_indices) * target_ratio)]
        train_sampled_indices = train_indices[sampled_indices]
        train_mask_target = np.zeros_like(train_mask)
        train_mask_target[train_sampled_indices] = True

        sampled_indices = np.random.permutation(len(val_indices))[:int(len(val_indices) * target_ratio)]
        val_sampled_indices = val_indices[sampled_indices]
        val_mask_target = np.zeros_like(val_mask)
        val_mask_target[val_sampled_indices] = True

        sampled_indices = np.random.permutation(len(test_indices))[:int(len(test_indices) * target_ratio)]
        test_sampled_indices = test_indices[sampled_indices]
        test_mask_target = np.zeros_like(test_mask)
        test_mask_target[test_sampled_indices] = True

        save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path2 = f'./processed_data/{dataset_name}/val_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path3 = f'./processed_data/{dataset_name}/test_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'

        np.save(save_file_path1, train_mask_target)
        np.save(save_file_path2, val_mask_target)
        np.save(save_file_path3, test_mask_target)
    else:
        save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path2 = f'./processed_data/{dataset_name}/val_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path3 = f'./processed_data/{dataset_name}/test_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'

        train_mask_target = np.load(save_file_path1)
        val_mask_target = np.load(save_file_path2)
        test_mask_target = np.load(save_file_path3)

    # graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    # edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    # node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                     edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask_target], dst_node_ids=dst_node_ids[train_mask_target],
                      node_interact_times=node_interact_times[train_mask_target],
                      edge_ids=edge_ids[train_mask_target], labels=labels[train_mask_target])
    val_data = Data(src_node_ids=src_node_ids[val_mask_target], dst_node_ids=dst_node_ids[val_mask_target],
                    node_interact_times=node_interact_times[val_mask_target], edge_ids=edge_ids[val_mask_target],
                    labels=labels[val_mask_target])
    test_data = Data(src_node_ids=src_node_ids[test_mask_target], dst_node_ids=dst_node_ids[test_mask_target],
                     node_interact_times=node_interact_times[test_mask_target], edge_ids=edge_ids[test_mask_target],
                     labels=labels[test_mask_target])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data


def get_node_classification_data_withtarget_sametest(dataset_name: str, val_ratio: float, test_ratio: float,
                                                     target_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
    if os.path.exists(save_file_path1) is False:
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        sampled_indices = np.random.permutation(len(train_indices))[:int(len(train_indices) * target_ratio)]
        train_sampled_indices = train_indices[sampled_indices]
        train_mask_target = np.zeros_like(train_mask)
        train_mask_target[train_sampled_indices] = True

        sampled_indices = np.random.permutation(len(val_indices))[:int(len(val_indices) * target_ratio)]
        val_sampled_indices = val_indices[sampled_indices]
        val_mask_target = np.zeros_like(val_mask)
        val_mask_target[val_sampled_indices] = True

        sampled_indices = np.random.permutation(len(test_indices))[:int(len(test_indices) * target_ratio)]
        test_sampled_indices = test_indices[sampled_indices]
        test_mask_target = np.zeros_like(test_mask)
        test_mask_target[test_sampled_indices] = True

        save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path2 = f'./processed_data/{dataset_name}/val_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path3 = f'./processed_data/{dataset_name}/test_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'

        np.save(save_file_path1, train_mask_target)
        np.save(save_file_path2, val_mask_target)
        np.save(save_file_path3, test_mask_target)
    else:
        save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path2 = f'./processed_data/{dataset_name}/val_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path3 = f'./processed_data/{dataset_name}/test_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'

        train_mask_target = np.load(save_file_path1)
        val_mask_target = np.load(save_file_path2)
        test_mask_target = np.load(save_file_path3)

    # graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    # edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    # node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                     edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask_target], dst_node_ids=dst_node_ids[train_mask_target],
                      node_interact_times=node_interact_times[train_mask_target],
                      edge_ids=edge_ids[train_mask_target], labels=labels[train_mask_target])
    val_data = Data(src_node_ids=src_node_ids[val_mask_target], dst_node_ids=dst_node_ids[val_mask_target],
                    node_interact_times=node_interact_times[val_mask_target], edge_ids=edge_ids[val_mask_target],
                    labels=labels[val_mask_target])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],
                     labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data


def get_node_classification_data_withtarget_ratio(dataset_name: str, val_ratio: float, test_ratio: float,
                                                  target_ratio: float):
    """
    generate data for node classification task on specific ratio
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    save_file_path1 = f'./processed_data/{dataset_name}/test_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
    if os.path.exists(save_file_path1) is False:
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        sampled_indices = np.random.permutation(len(train_indices))[:int(len(train_indices) * target_ratio)]
        train_sampled_indices = train_indices[sampled_indices]
        train_mask_target = np.zeros_like(train_mask)
        train_mask_target[train_sampled_indices] = True

        sampled_indices = np.random.permutation(len(val_indices))[:int(len(val_indices) * target_ratio)]
        val_sampled_indices = val_indices[sampled_indices]
        val_mask_target = np.zeros_like(val_mask)
        val_mask_target[val_sampled_indices] = True

        sampled_indices = np.random.permutation(len(test_indices))[:int(len(test_indices) * target_ratio)]
        test_sampled_indices = test_indices[sampled_indices]
        test_mask_target = np.zeros_like(test_mask)
        test_mask_target[test_sampled_indices] = True

        save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path2 = f'./processed_data/{dataset_name}/val_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path3 = f'./processed_data/{dataset_name}/test_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'

        np.save(save_file_path1, train_mask_target)
        np.save(save_file_path2, val_mask_target)
        np.save(save_file_path3, test_mask_target)
    else:
        save_file_path1 = f'./processed_data/{dataset_name}/train_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path2 = f'./processed_data/{dataset_name}/val_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'
        save_file_path3 = f'./processed_data/{dataset_name}/test_mask_{dataset_name}_withtarget_{str(target_ratio)}.npy'

        train_mask_target = np.load(save_file_path1)
        val_mask_target = np.load(save_file_path2)
        test_mask_target = np.load(save_file_path3)

    # graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    # edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    # node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                     edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask_target], dst_node_ids=dst_node_ids[train_mask_target],
                      node_interact_times=node_interact_times[train_mask_target],
                      edge_ids=edge_ids[train_mask_target], labels=labels[train_mask_target])
    val_data = Data(src_node_ids=src_node_ids[val_mask_target], dst_node_ids=dst_node_ids[val_mask_target],
                    node_interact_times=node_interact_times[val_mask_target], edge_ids=edge_ids[val_mask_target],
                    labels=labels[val_mask_target])
    test_data = Data(src_node_ids=src_node_ids[test_mask_target], dst_node_ids=dst_node_ids[test_mask_target],
                     node_interact_times=node_interact_times[test_mask_target], edge_ids=edge_ids[test_mask_target],
                     labels=labels[test_mask_target])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data


def get_node_classification_dgraph(dataset_name: str, val_ratio: float, test_ratio: float,
                                   target_ratio: float):
    """
    generate data for node classification task on dgraph
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[
        1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[
        1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
        1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    root = "/data/renhong/dyg"
    save_file_path1 = root + f'/processed_data/dgraph/train_mask_dgraph.npy'
    save_file_path2 = root + f'/processed_data/dgraph/val_mask_dgraph.npy'
    save_file_path3 = root + f'/processed_data/dgraph/test_mask_dgraph.npy'

    train_mask_target = np.load(save_file_path1, allow_pickle=True)
    val_mask_target = np.load(save_file_path2, allow_pickle=True)
    test_mask_target = np.load(save_file_path3, allow_pickle=True)

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                     edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask_target], dst_node_ids=dst_node_ids[train_mask_target],
                      node_interact_times=node_interact_times[train_mask_target],
                      edge_ids=edge_ids[train_mask_target], labels=labels[train_mask_target])
    val_data = Data(src_node_ids=src_node_ids[val_mask_target], dst_node_ids=dst_node_ids[val_mask_target],
                    node_interact_times=node_interact_times[val_mask_target], edge_ids=edge_ids[val_mask_target],
                    labels=labels[val_mask_target])
    test_data = Data(src_node_ids=src_node_ids[test_mask_target], dst_node_ids=dst_node_ids[test_mask_target],
                     node_interact_times=node_interact_times[test_mask_target], edge_ids=edge_ids[test_mask_target],
                     labels=labels[test_mask_target])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data


def calculate_pagerank(edge_index, damping=0.85, max_iter=50, tol=1e-6):
    u, v = edge_index
    num_nodes = np.max(edge_index) + 1
    in_adj = [[] for _ in range(num_nodes)]
    out_degree = np.zeros(num_nodes, dtype=int)
    for src, dst in zip(u, v):
        in_adj[dst].append(src)
        out_degree[src] += 1

    scores = np.ones(num_nodes) / num_nodes

    for _ in range(max_iter):
        new_scores = np.full(num_nodes, (1 - damping) / num_nodes)
        for dst in range(num_nodes):
            for src in in_adj[dst]:
                if out_degree[src] > 0:
                    new_scores[dst] += damping * scores[src] / out_degree[src]

        if np.linalg.norm(new_scores - scores) < tol:
            break
        scores = new_scores
    return scores



def calculate_temporal_pagerank(edge_index, edge_timestamp, damping=0.85, lambda_=0.1, max_iter=50, tol=1e-6):
    """
    结合时间戳的PageRank计算（边权重随时间衰减）

    参数:
    edge_index (np.ndarray): 形状为(2, M)的边列表（源/目标节点）
    edge_timestamp (np.ndarray): 形状为(M,)的时间戳数组（与边一一对应）
    damping (float): 阻尼因子，默认0.85
    lambda_ (float): 时间衰减因子（越大则早期边权重衰减越快）
    max_iter (int): 最大迭代次数
    tol (float): 收敛阈值

    返回:
    np.ndarray: 节点PageRank得分数组
    """
    u, v = edge_index
    M = u.shape[0]
    num_nodes = np.max(edge_index) + 1

    in_adj = [[] for _ in range(num_nodes)]
    out_degree = np.zeros(num_nodes, dtype=int)
    current_time = np.max(edge_timestamp)

    for i in range(M):
        src, dst = u[i], v[i]
        timestamp = edge_timestamp[i]
        in_adj[dst].append((src, timestamp))
        out_degree[src] += 1

    scores = np.ones(num_nodes) / num_nodes

    for _ in tqdm.tqdm(range(max_iter)):
        new_scores = np.full(num_nodes, (1 - damping) / num_nodes)

        for dst in range(num_nodes):
            total_weight = 0.0
            for (src, t) in in_adj[dst]:
                if out_degree[src] == 0:
                    continue
                time_diff = current_time - t
                edge_weight = np.exp(-lambda_ * time_diff)
                contribution = scores[src] * edge_weight / out_degree[src]
                new_scores[dst] += damping * contribution

        if np.linalg.norm(new_scores - scores) < tol:
            break
        scores = new_scores

    return scores
