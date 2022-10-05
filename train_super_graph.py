import argparse
import json, re, sys, os, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
# import pickle as pkl
import dgl
import tensorflow as tf
from tensorflow.keras import layers
from dgl.nn.tensorflow import GraphConv
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import psutil


def get_mem_used():
    return psutil.virtual_memory().used / 2**20

class GCN(tf.keras.Model):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layer_list = []
        # input layer
        self.layer_list.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layer_list.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layer_list.append(GraphConv(n_hidden, n_classes))
        self.dropout = layers.Dropout(dropout)

    def call(self, features):
        h = features
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
    
    # def get_embedding(self, features):
    #     h = features
    #     for i, layer in enumerate(self.layer_list[:-1]):
    #         # if i != 0:
    #         #     h = self.dropout(h)
    #         h = layer(self.g, h)
    #     return h

def convert_dict_to_int(d):
    assert type(d) == dict
    new_d = {}
    for k,v in d.items():
        if type(v) == list:
            new_d[int(k)] = [int(i) for i in v]
        else:
            new_d[int(k)] = int(v)
    return new_d

def get_tuple_key_dict(in_dict):
    out_dict = {}
    for k, v in in_dict.items():
        new_k = tuple([int(x.strip()) for x in str(k).strip('(').strip(')').split(',')])
        assert len(new_k) == 2
        out_dict[new_k] = v
    return out_dict

def generate_masks(n_data, test_size=0.2):
    X = np.array(range(n_data))
    train, other, _, _ = train_test_split(X, X, test_size=test_size)
    val, test, _, _ = train_test_split(X[other], X[other], test_size=0.5)
    train_mask = np.array([False] * n_data)
    val_mask = np.array([False] * n_data)
    test_mask = np.array([False] * n_data)
    train_mask[train] = True
    val_mask[val] = True
    test_mask[test] = True
    return (
        tf.convert_to_tensor(train_mask, dtype='bool'),
        tf.convert_to_tensor(val_mask, dtype='bool'), 
        tf.convert_to_tensor(test_mask, dtype='bool')
    )

def generate_masks_from_idx(X, n_data, test_size=0.2):
    train, other, _, _ = train_test_split(X, X, test_size=test_size)
    val, test, _, _ = train_test_split(other, other, test_size=0.5)
    train_mask = np.array([False] * n_data)
    val_mask = np.array([False] * n_data)
    test_mask = np.array([False] * n_data)
    train_mask[train] = True
    val_mask[val] = True
    test_mask[test] = True
    return (
        tf.convert_to_tensor(train_mask, dtype='bool'),
        tf.convert_to_tensor(val_mask, dtype='bool'), 
        tf.convert_to_tensor(test_mask, dtype='bool')
    )

def generate_mask_from_idx(idx, n_nodes):
    mask = [False] * n_nodes
    for i in idx:
        mask[i] = True
    return tf.convert_to_tensor(mask, dtype='bool')

def get_train_val_test_split(args, community2node, original_test_mask, n_communities):
    if args.overlap:
        # TODO: CPM only add communities which is not in valid/test of original graph
        # because many nodes do not have community membership and
        # these nodes are assigned to a new community of themselves alone
        print('CPM train-val-test split')
        # one original node is itself a community: len(community2node[community]) == 1
        # and it is in the original test data: sum of corresponding positions of the original test mask is not 0
        # hence it must be test data
        test_communities = [community for community in community2node if len(community2node[community]) == 1 and tf.reduce_sum(tf.gather(tf.cast(original_test_mask, dtype=tf.float32), community2node[community])).numpy().item() > 0.5]
        train_communities = [community for community in community2node if len(community2node[community]) > 1]
        # communities that can be used for training
        available_communities = list(set(range(n_communities)).difference(set(test_communities)))
        print("num of... | communities: {} | must be test: {} | available for training: {}".format(n_communities, len(test_communities), len(available_communities)))
        # get split
        if args.test_split_ratio == 0.0:
            print('Using all available communities for training')
            return (
                generate_mask_from_idx(available_communities, n_communities),
                generate_mask_from_idx(available_communities, n_communities),
                generate_mask_from_idx(available_communities, n_communities)
            )
            # super_graph.ndata['train_mask'] = generate_mask_from_idx(available_communities, n_communities)
            # super_graph.ndata['val_mask'] = generate_mask_from_idx(available_communities, n_communities)
            # super_graph.ndata['test_mask'] = generate_mask_from_idx(available_communities, n_communities)
        else:
            print('Use {:.2%} of all available communities for training'.format(1 - args.test_split_ratio))
            train_mask_, val_mask_, test_mask_ = generate_masks_from_idx(available_communities, n_communities, test_size=args.test_split_ratio)
            # TODO: optional: add "real" communities to train data, need to handle data between cpu and gpu
            # train_mask_ = tf.where(generate_mask_from_idx(train_communities, n_communities), [True] * n_communities, train_mask_)
            # TODO: optional: add test_communities to the test data, need to handle data between cpu and gpu
            # test_mask_ = tf.logical_or(test_mask_, generate_mask_from_idx(test_communities, n_communities))
            return (train_mask_, val_mask_, test_mask_)
            # super_graph.ndata['train_mask'] = tf.convert_to_tensor(train_mask_, dtype='bool')
            # super_graph.ndata['val_mask'] = tf.convert_to_tensor(val_mask_, dtype='bool')
            # super_graph.ndata['test_mask'] = tf.convert_to_tensor(test_mask_, dtype='bool')
    else:
        # LPA just split without taking consideration of test cases because no node is itself a community
        print('LPA train-val-test split')
        if args.test_split_ratio == 0.0:
            print('Using all communities for training')
            return (
                tf.convert_to_tensor([True] * n_communities, dtype='bool'),
                tf.convert_to_tensor([True] * n_communities, dtype='bool'),
                tf.convert_to_tensor([True] * n_communities, dtype='bool')
            )
            # super_graph.ndata['train_mask'] = tf.convert_to_tensor([True] * n_communities, dtype='bool')
            # super_graph.ndata['val_mask'] = tf.convert_to_tensor([True] * n_communities, dtype='bool')
            # super_graph.ndata['test_mask'] = tf.convert_to_tensor([True] * n_communities, dtype='bool')
        else:
            print('Use {:.2%} of all communities for training'.format(1 - args.test_split_ratio))
            return generate_masks(n_data=n_communities, test_size=args.test_split_ratio)
            # train_mask_, val_mask_, test_mask_ = generate_masks(n_data=n_communities, test_size=args.test_split_ratio)
            # return (
            #     tf.convert_to_tensor(train_mask_, dtype='bool'),
            #     tf.convert_to_tensor(val_mask_, dtype='bool'),
            #     tf.convert_to_tensor(test_mask_, dtype='bool')
            # )
            # super_graph.ndata['train_mask'] = tf.convert_to_tensor(train_mask_, dtype='bool')
            # super_graph.ndata['val_mask'] = tf.convert_to_tensor(val_mask_, dtype='bool')
            # super_graph.ndata['test_mask'] = tf.convert_to_tensor(test_mask_, dtype='bool')


def evaluate(model, features, labels, mask):
    logits = model(features, training=False)
    logits = logits[mask]
    labels = labels[mask]
    labels = tf.math.argmax(labels, axis=1) # TODO: delete if use keras loss
    indices = tf.math.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
    return acc.numpy().item()

def evaluate_original_tf(model, features, labels, mask, n_nodes, node2community, overlapping=False):
    """
    cacluate accuracy by tf
    input:
        features: of super graph
        labels: of original graph
        mask: of original graph
        n_nodes: of original graph
    """
    logits = model(features, training=False)
    community_pred = tf.math.argmax(logits, axis=1)
    
    test_idx = np.array(range(n_nodes))[mask]
    n_test_nodes = len(test_idx)

    valid_test_idx = [node for node in test_idx if node in node2community]
    y_true = tf.gather(labels, valid_test_idx) # labels[valid_test_idx]
    num_valid = len(valid_test_idx)

    if overlapping:
        y_pred = [tf.reduce_sum(tf.gather(logits, node2community[node]), axis=0) for node in valid_test_idx]
        y_pred = tf.math.argmax(y_pred, axis=1)
    else:
        y_pred = [community_pred[node2community[node]] for node in valid_test_idx]
    
    num_correct = tf.reduce_sum(tf.cast(y_true == y_pred, dtype=tf.float32)).numpy().item()
    acc = num_correct / num_valid    # acc = tf.reduce_mean(tf.cast(y_true == y_pred), dtype=tf.float32))
    print("Total test data: {} | valid test data: {} | valid percent {:.2%} | correct {}".format(
        n_test_nodes, num_valid, num_valid/n_test_nodes, num_correct))
    print("Test Accuracy on original graph {:.4f}".format(acc))

    return acc

# TODO: remove input arg community2node
def evaluate_original(model, features, labels, mask, n_nodes, node2community, community2node, overlapping=False):
    """
    Inputs
        n_nodes: number of nodes of the original graph
        mask: test mask of the original graph 
        labels: true labels of the original graph, shape = (number of test cases, )
    Note "labels" is NOT one-hot
    """
    # cacluate accuracy by counter
    logits = model(features, training=False)
    community_pred = tf.math.argmax(logits, axis=1)
    
    test_idx = np.array(range(n_nodes))[mask]
    n_test_nodes = len(test_idx)

    community_stat = np.zeros((len(community2node), 2)) # [ [# test, # correct] ]

    num_correct = 0
    num_valid = 0
    for node in tqdm(test_idx):
        if node in node2community:
            num_valid += 1
            community_stat[node2community[node]][0] += 1 
            if overlapping:
                if labels[node] == tf.math.argmax(tf.reduce_sum(tf.gather(logits, node2community[node]), axis=0), axis=0):
                    num_correct += 1
                    community_stat[node2community[node]][1] += 1 
            elif labels[node] == community_pred[node2community[node]]:
                num_correct += 1
                community_stat[node2community[node]][1] += 1 
    acc = num_correct / num_valid
    print("Total test data: {} | valid test data: {} | valid precent {:2%} | correct {}".format(
        n_test_nodes, num_valid, num_valid/n_test_nodes, num_correct))
    print("Test Accuracy on original graph {:.4f}".format(acc))
    
    # TODO: delete
    df_data = []
    for c in range(len(set(community2node.keys()))):
        df_data.append([c, len(community2node[c]), community_stat[c][0], community_stat[c][1], 
            np.round(community_stat[c][1] / community_stat[c][0], 4)])
    pd.DataFrame(df_data, columns=['community_id', 'nodes', 'test', 'correct', 'accuracy']).to_csv('community_stat.csv', index=False)
    
    return acc

# # wrong version
# def evaluate_original(model, features, labels, mask, n_nodes, node2community):
#     # cacluate accuracy by counter
#     logits = model(features, training=False)
#     indices_true = tf.math.argmax(labels, axis=1)
#     indices_pred = tf.math.argmax(logits, axis=1)
    
#     test_idx = np.array(range(n_nodes))[mask]
#     n_test_nodes = len(test_idx)

#     num_correct = 0
#     num_valid = 0
#     for node in tqdm(test_idx):
#         if node in node2community:
#             num_valid += 1
#             if indices_true[node2community[node]] == indices_pred[node2community[node]]:
#                 num_correct += 1
#     acc = num_correct / num_valid
#     print("Total test data: {:d} | valid test data: {:d} | valid precent {:2f} | correct {:d}".format(
#         n_test_nodes, num_valid, num_valid/n_test_nodes, num_correct))
#     print("Test Accuracy on original graph {:.4f}".format(acc))
#     return acc

# # wrong version
# def evaluate_original_tf(model, features, labels, mask, n_nodes, n_communities, community2node):
#     # calcuate accuracy by tf.reduce
#     logits = model(features, training=False)
#     indices_true = tf.math.argmax(labels, axis=1)
#     indices_pred = tf.math.argmax(logits, axis=1)
#     # for nodes
#     y_true = np.zeros(n_nodes, dtype='int64')
#     y_pred = np.zeros(n_nodes, dtype='int64')
#     for i in tqdm(range(n_communities)):
#         for node in community2node[i]:
#             y_true[node] = indices_true[i]
#             y_pred[node] = indices_pred[i]
#     # acc = tf.reduce_mean(tf.cast(y_pred == y_true, dtype=tf.float32)).numpy().item()
#     # print("All Accuracy on original graph {:.4f}".format(acc))
#     # test nodes
#     # test_mask = graph.ndata['test_mask']# .numpy()
#     y_true = y_true[mask]
#     y_pred = y_pred[mask]
#     acc = tf.reduce_mean(tf.cast(y_pred == y_true, dtype=tf.float32)).numpy().item()
#     print("Test Accuracy on original graph {:.4f}".format(acc))
#     return acc

def load_data(args):
    if args.dataset == 'ogb':
        # TODO: bug with DGL: dgl._ffi.base.DGLError: Cannot assign node feature "label" on device /gpu:0 to a graph on device /cpu.
        # Call DGLGraph.to()
        # dataset = DglNodePropPredDataset(name='ogbn-products', root=args.ogb_dir)
        # graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        # graph.ndata['label'] = tf.reshape(label, [-1])
        # # graph.ndata['label'] = np.array(label, dtype='int32').flatten()
        # # masks
        # split_idx = dataset.get_idx_split()
        # train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # n_nodes = graph.num_nodes()
        # graph.ndata['train_mask'] = generate_mask_from_idx(train_idx, n_nodes)
        # graph.ndata['val_mask'] = generate_mask_from_idx(val_idx, n_nodes)
        # graph.ndata['test_mask'] = generate_mask_from_idx(test_idx, n_nodes)
        dataset = DglNodePropPredDataset(name='ogbn-products', root=args.ogb_dir)
        graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        n_nodes = graph.num_nodes()
        graph.ndata['train_mask'] = generate_mask_from_idx(train_idx, n_nodes)
        graph.ndata['val_mask'] = generate_mask_from_idx(val_idx, n_nodes)
        graph.ndata['test_mask'] = generate_mask_from_idx(test_idx, n_nodes)
        if args.gpu >= 0:
            graph = graph.to('/gpu:0') # TODO: get label's device
        graph.ndata['label'] = tf.reshape(label, [-1])
        print('label shape: {}'.format(graph.ndata['label'].shape))
        if args.gpu >= 0:
            graph = graph.to('/cpu:0') # TODO: check device
        # original_test_idx = test_idx
    else:
        # dataset from DGL
        if args.dataset.lower().startswith('reddit'):
            dataset = dgl.data.RedditDataset(raw_dir=args.dgl_dir)
        elif args.dataset.lower().startswith('cora'):
            dataset = dgl.data.CoraGraphDataset(raw_dir=args.dgl_dir)
        elif args.dataset.lower().startswith('pubmed'):
            dataset = dgl.data.PubmedGraphDataset(raw_dir=args.dgl_dir)
        elif args.dataset.lower().startswith('citeseer'):
            dataset = dgl.data.CiteseerGraphDataset(raw_dir=args.dgl_dir)
        else:
            raise NotImplementedError()
        graph = dataset[0]
        # TODO: uncomment if use FastGCN version of pubmed
        # if args.dataset.lower().startswith('pubmed'):
        #     graph.ndata['train_mask'] = tf.convert_to_tensor([True] * 18217 + [False] * 500 + [False] * 1000, dtype='bool')
        #     graph.ndata['val_mask'] = tf.convert_to_tensor([False] * 18217 + [True] * 500 + [False] * 1000, dtype='bool')
        #     graph.ndata['test_mask'] = tf.convert_to_tensor([False] * 18217 + [False] * 500 + [True] * 1000, dtype='bool')
    n_classes = dataset.num_classes
    return graph, n_classes

def main(args):
    # load dataset
    print('loading dataset...')
    graph, n_classes = load_data(args)
    # store some information
    original_n_nodes = graph.number_of_nodes()
    # original_test_idx = np.array(range(original_n_nodes))[graph.ndata['test_mask']]
    original_test_mask = graph.ndata['test_mask']
    original_labels = graph.ndata['label']
    print('Original graph has {} nodes and {} test data.'.format(original_n_nodes, np.sum(original_test_mask)))

    # load community info
    print('loading community info...')
    with open(os.path.join(args.src_dir, args.dataset + '_node2community.json'), 'r') as f:
        node2community = convert_dict_to_int(json.load(f))
    with open(os.path.join(args.src_dir, args.dataset + '_community2node.json'), 'r') as f:
        community2node = convert_dict_to_int(json.load(f))
    with open(os.path.join(args.src_dir, args.dataset + '_sim.json'), 'r') as f:
        sim = get_tuple_key_dict(json.load(f))
    # load super graph data
    super_graph_data = np.load(os.path.join(args.src_dir, args.dataset + '.npz'))
    np_new_feats = super_graph_data['feat'] # np.load(os.path.join(args.src_dir, args.dataset + '_new_feats.npz'))['data']
    np_new_labels = super_graph_data['label'] # np.load(os.path.join(args.src_dir, args.dataset + '_new_labels.npz'))['data']

    if args.threshold_val <= 0:
        sim_vals_sorted = np.load(os.path.join(args.src_dir, args.dataset + '_sim_val.npz'))['data']
        #threshold = sim_vals_sorted[int(len(sim_vals_sorted) / 2)]
        threshold = sim_vals_sorted[args.threshold_id]
    else:
        threshold = args.threshold_val
    n_communities = len(set(community2node.keys())) # n_communities = len(set(node2community.values()))
    print('n_communities = {}, threshold = {}'.format(n_communities, threshold))

    # get super graph
    print('creating the contracted graph...')
    super_graph = nx.DiGraph()
    super_graph.add_nodes_from(range(n_communities))
    super_graph.add_edges_from([edge for edge in sim if sim[edge] >= threshold])
    super_graph = dgl.convert.from_networkx(super_graph)
    # ['feat', 'label', 'train_mask', 'val_mask', 'test_mask']
    super_graph.ndata['feat'] = tf.convert_to_tensor(np_new_feats, dtype='float32')
    super_graph.ndata['label'] = tf.convert_to_tensor(np_new_labels, dtype='float32')
    # train-val-test split
    super_graph.ndata['train_mask'], \
        super_graph.ndata['val_mask'], \
        super_graph.ndata['test_mask'] \
        = get_train_val_test_split(args, community2node, original_test_mask, n_communities)
    # # The below train-val-test split codes are put into the function get_train_val_test_split()
    # if args.overlap:
    #     # TODO: CPM only add communities which is not in valid/test of original graph
    #     # because many nodes do not have community membership and
    #     # these nodes are assigned to a new community of themselves alone
    #     print('CPM train-val-test split')
    #     # one original node is itself a community: len(community2node[community]) == 1
    #     # and it is in the original test data: sum of corresponding positions of the original test mask is not 0
    #     # hence it must be test data
    #     test_communities = [community for community in community2node if len(community2node[community]) == 1 and tf.reduce_sum(tf.gather(tf.cast(original_test_mask, dtype=tf.float32), community2node[community])).numpy().item() > 0.5]
    #     # communities that can be used for training
    #     available_communities = list(set(range(n_communities)).difference(set(test_communities)))
    #     print("num of... | communities: {} | must be test: {} | available for training: {}".format(n_communities, len(test_communities), len(available_communities)))
    #     # get split
    #     if args.test_split_ratio == 0.0:
    #         print('Using all available communities for training')
    #         super_graph.ndata['train_mask'] = generate_mask_from_idx(available_communities, n_communities)
    #         super_graph.ndata['val_mask'] = generate_mask_from_idx(available_communities, n_communities)
    #         super_graph.ndata['test_mask'] = generate_mask_from_idx(available_communities, n_communities)
    #     else:
    #         pass
    #         print('Use {:.2%} of all available communities for training'.format(1 - args.test_split_ratio))
    #         train_mask_, val_mask_, test_mask_ = generate_masks_from_idx(available_communities, n_communities, test_size=args.test_split_ratio)
    #         # TODO: optional: add test_communities to the test data
    #         # test_mask_ = tf.logical_or(test_mask_, generate_mask_from_idx(test_communities, n_communities)) 
    #         super_graph.ndata['train_mask'] = tf.convert_to_tensor(train_mask_, dtype='bool')
    #         super_graph.ndata['val_mask'] = tf.convert_to_tensor(val_mask_, dtype='bool')
    #         super_graph.ndata['test_mask'] = tf.convert_to_tensor(test_mask_, dtype='bool')
    # else:
    #     # LPA just split without taking consideration of test cases because no node is itself a community
    #     print('LPA train-val-test split')
    #     if args.test_split_ratio == 0.0:
    #         print('Using all communities for training')
    #         super_graph.ndata['train_mask'] = tf.convert_to_tensor([True] * n_communities, dtype='bool')
    #         super_graph.ndata['val_mask'] = tf.convert_to_tensor([True] * n_communities, dtype='bool')
    #         super_graph.ndata['test_mask'] = tf.convert_to_tensor([True] * n_communities, dtype='bool')
    #     else:
    #         print('Use {:.2%} of all communities for training'.format(1 - args.test_split_ratio))
    #         train_mask_, val_mask_, test_mask_ = generate_masks(n_data=n_communities, test_size=args.test_split_ratio)
    #         super_graph.ndata['train_mask'] = tf.convert_to_tensor(train_mask_, dtype='bool')
    #         super_graph.ndata['val_mask'] = tf.convert_to_tensor(val_mask_, dtype='bool')
    #         super_graph.ndata['test_mask'] = tf.convert_to_tensor(test_mask_, dtype='bool')

    # show super graph information
    print(super_graph)
    
    g = super_graph
    if args.gpu < 0:
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(args.gpu)
        g = g.to(device)
    
    # training
    print('training...')
    with tf.device(device):
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        # n_classes = data.num_labels
        n_edges = g.number_of_edges()
        print("""----Data statistics------'
        #Edges %d
        #Classes %d
        #Train samples %d
        #Val samples %d
        #Test samples %d""" %
            (n_edges, n_classes,
            train_mask.numpy().sum(),
            val_mask.numpy().sum(),
            test_mask.numpy().sum()))

        # add self loop
        if args.self_loop:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)

        n_edges = g.number_of_edges()
        # normalization
        degs = tf.cast(tf.identity(g.in_degrees()), dtype=tf.float32)
        norm = tf.math.pow(degs, -0.5)
        norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)

        g.ndata['norm'] = tf.expand_dims(norm, -1)

        model = GCN(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    tf.nn.relu,
                    args.dropout)

        # loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # TODO: uncomment if use keras loss 
        # use optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr, epsilon=1e-8)

        # initialize graph
        best_val_test = 0.0
        dur = []
        max_mem_usage, cur_mem_usage = 0., 0.
        # labels = tf.one_hot(labels, n_classes) # TODO: delete if use keras loss 
        for epoch in range(args.n_epochs):
            if epoch >= 0:#3:
                t0 = time.time()
            m0 = get_mem_used()
            # forward
            with tf.GradientTape() as tape:
                logits = model(features)
                # loss_value = loss_fcn(labels[train_mask], logits[train_mask])
                loss_value = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[train_mask], logits=logits[train_mask]))

                # Manually Weight Decay
                # We found Tensorflow has a different implementation on weight decay
                # of Adam(W) optimizer with PyTorch. And this results in worse results.
                # Manually adding weights to the loss to do weight decay solves this problem.
                for weight in model.trainable_weights:
                    loss_value = loss_value + \
                        args.weight_decay*tf.nn.l2_loss(weight)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            cur_mem_usage = get_mem_used() - m0
            if cur_mem_usage > max_mem_usage:
                max_mem_usage = cur_mem_usage
            if epoch >= 0:# 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, features, labels, val_mask)
            # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            #         "ETputs(KTEPS) {:.2f} | Time(s) {:.4f}".format(epoch, np.mean(dur), loss_value.numpy().item(),
            #                                     acc, n_edges / np.mean(dur) / 1000, np.mean(dur)))

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | ETputs(KTEPS) {:.2f} | "
                "Time(s) {:.4f} | CPU(MB) {:.2f} | GPU(MB) {:.2f}".format(epoch, np.mean(dur), loss_value.numpy().item(),
                acc, n_edges / np.mean(dur) / 1000, np.mean(dur), max_mem_usage, 0))
                
            # # save weights
            # if acc > best_val_test and args.save_model is True:
            #     model.save_weights('models/{}/{}/'.format(args.dataset, 'thresholdID_' + str(args.threshold_id)))

        acc = evaluate(model, features, labels, test_mask)
        print("Test Accuracy on contracted graph {:.4f}\n".format(acc))

        if args.save_model is True:
            model.save_weights('models/{}/{}{:.4f}/'.format(args.dataset, 'thresholdID_' + str(args.threshold_id) + '_acc_', acc))
        # # wrong implementations
        # acc_tf = evaluate_original_tf(model, features, labels, original_test_mask, original_n_nodes, n_communities, community2node)
        # acc = evaluate_original(model, features, labels, original_test_mask, original_n_nodes, node2community)
        
        t0 = time.time()
        acc = evaluate_original_tf(model, features, original_labels, original_test_mask, original_n_nodes, node2community, args.overlap)
        # acc = evaluate_original(model, features, original_labels, original_test_mask, original_n_nodes, node2community, community2node, args.overlap)
        t1 = time.time()
        print("Test acc: {:.2%} | Test time: {:.4f} seconds, i.e. {:.4f} minutes".format(acc, t1 - t0, (t1 - t0) / 60))
        
        # # slower implementations
        # t0 = time.time()
        # acc = evaluate_original(model, features, original_labels, original_test_mask, original_n_nodes, node2community, args.overlap)
        # t1 = time.time()
        # print("Test acc: {:.2%} | Test time: {:.4f} seconds, i.e. {:.4f} minutes".format(acc, t1 - t0, (t1 - t0) / 60))
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on super graph")
    parser.add_argument("--dataset", type=str, required=True,
        help= "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=True)")
    parser.set_defaults(self_loop=True)
    # required args
    parser.add_argument("--threshold-val", type=float, default=-1)
    parser.add_argument("--threshold-id", type=int, default=600,
                        help="index of similarity score for the threshold for adding edges between communities")
    parser.add_argument("--test-split-ratio", type=float, default=0,
                        help="ratio for valiation and test data")
    parser.add_argument("--overlap", type=bool, default=False,
                        help="whether the community is overlapping")
    parser.add_argument("--src-dir", type=str, default="LabelPropagation/output",
                        help="source directory to retrieve super graph")
    # default args
    parser.add_argument("--dgl-dir", type=str, default="/export/data/zhiyuan/.dgl",
                        help="source directory which stores DGL data")
    parser.add_argument("--ogb-dir", type=str, default="/export/data/zhiyuan/CommunityGCN/OGB/",
                        help="source directory which stores OGB data")
    parser.add_argument("--save-model", type=bool, default=True,
                        help="whether save model to disk")
    args = parser.parse_args()
    print(args)

    main(args)

    my_pid = os.getpid()
    print(os.system('grep VmPeak /proc/' + str(my_pid) + '/status'))
    print(os.system('grep VmHWM /proc/' + str(my_pid) + '/status'))