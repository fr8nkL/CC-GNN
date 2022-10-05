import argparse
import json, re, sys, os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
# import pickle as pkl
import dgl
# from dgl import DGLGraph
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
from ogb.nodeproppred import DglNodePropPredDataset # Load Node Property Prediction datasets in OGB

def encode_onehot(labels, n_classes=None):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

# def encode_onehot(labels, n_classes=None):
#     if n_classes is None:
#         n_classes = len(set(labels))
#     labels = np.array(labels)
#     n_data = len(labels)
#     assert len(labels.shape) == 1
#     result = np.zeros(shape=(n_data, n_classes))
#     for i in range(n_data):
#         result[i][label[i] % n_classes] = 1.0
#     return result

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
    return train_mask, val_mask, test_mask

def generate_mask_from_idx(idx, n_nodes):
    mask = [False] * n_nodes
    for i in idx:
        mask[i] = True
    return tf.convert_to_tensor(mask, dtype='bool')
    
def convert_dict_to_int(d):
    assert type(d) == dict
    new_d = {}
    for k,v in tqdm(d.items()):
        if type(v) == list:
            new_d[int(k)] = [int(i) for i in v]
        else:
            new_d[int(k)] = int(v)
    return new_d

def get_node_community_dict(lpa_output_path):
    '''
    return two dicts: node2community, community2node
    community values are normalized to range [0, n_communities)
    '''
    node2community = {}
    community2node = {}
    # get node2community
    with open(lpa_output_path, "r") as json_file:
        node2community = convert_dict_to_int(json.load(json_file))
    community_vals = list(set(node2community.values()))
    # community_vals.sort() # optional sort
    # normalize community vals to range [0, n_communities)
    community_idx_map = {}
    for i,j in tqdm(enumerate(community_vals)):
        community_idx_map[j] = i
    for node in tqdm(node2community.keys()):
        prev_community_num = node2community[node]
        node2community[node] = community_idx_map[prev_community_num]
    # get community2node
    for node, community in tqdm(node2community.items()):
        if community in community2node:
            community2node[community].append(node)
        else:
            community2node[community] = [node]
    print('num_communities = {} | num of nodes with community = {}'.format(
        len(set(node2community.values())), len(node2community.keys())))
    return node2community, community2node

def calculate_sim_scores(graph, node2community, community2node):
    # sim[(A,B)] := similarity between community A and B
    sim = {}
    # first calculate the number of inter-community edges
    src, dst = graph.edges()
    src, dst = src.numpy(), dst.numpy()
    for idx in tqdm(range(len(src))):
        edge = (src[idx], dst[idx])
        k = (node2community[edge[0]], node2community[edge[1]])
        if k[0] != k[1]:
            if k in sim:
                sim[k] += 1
            else:
                sim[k] = 1
    # calculate similariy
    union_size = {}
    community_set = {}
    for k in tqdm(sim.keys()):
        union_size_key = k if k[0] < k[1] else (k[1], k[0]) # small number first
        if union_size_key in union_size:
            denominator = union_size[union_size_key]
        else:
            if union_size_key[0] not in community_set:
                community_set[union_size_key[0]] = set(community2node[union_size_key[0]])
            if union_size_key[1] not in community_set:
                community_set[union_size_key[1]] = set(community2node[union_size_key[1]])
            denominator = len(community_set[union_size_key[0]].union(community_set[union_size_key[1]]))
            # denominator = len(set(community2node[union_size_key[0]]).union(set(community2node[union_size_key[1]])))
            union_size[union_size_key] = denominator
        sim[k] /= denominator
    return sim

def main(args):
    # load dataset
    if args.dataset.startswith('ogb'):
        dataset = DglNodePropPredDataset(name=args.dataset, root=args.ogb_dir)
        graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        graph.ndata['label'] = tf.reshape(label, [-1])
        # graph.ndata['label'] = np.array(label, dtype='int32').flatten()
        # masks
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        n_nodes = graph.num_nodes()
        graph.ndata['train_mask'] = generate_mask_from_idx(train_idx, n_nodes)
        graph.ndata['val_mask'] = generate_mask_from_idx(val_idx, n_nodes)
        graph.ndata['test_mask'] = generate_mask_from_idx(test_idx, n_nodes)
    else:
        # dataset from DGL
        if args.dataset.startswith('reddit'):
            dataset = dgl.data.RedditDataset(raw_dir=args.dgl_dir)
        elif args.dataset.startswith('cora'):
            dataset = dgl.data.CoraGraphDataset(raw_dir=args.dgl_dir)
        elif args.dataset.startswith('pubmed'):
            dataset = dgl.data.PubmedGraphDataset(raw_dir=args.dgl_dir)
        elif args.dataset.startswith('citeseer'):
            dataset = dgl.data.CiteseerGraphDataset(raw_dir=args.dgl_dir)
        else:
            raise NotImplementedError()
        graph = dataset[0]
    # ['feat', 'label', 'train_mask', 'val_mask', 'test_mask']
    # for key in ['feat', 'label']:
    #     print('shape of {} = {}'.format(key, graph.ndata[key].shape))
    # print('number of classes = {}'.format(dataset.num_classes))


    preprocess_start_time = time.time()
    print('processing community assignment...')
    lpa_output_path = os.path.join(args.output_dir, args.dataset + '.json') # 'output/cora.json'
    node2community, community2node = get_node_community_dict(lpa_output_path)
    n_communities = len(set(node2community.values()))
    # save node--community dicts to file
    node2community_path = os.path.join(args.output_dir, args.dataset + '_node2community.json') # 'output/cora_node2community.json'
    community2node_path = os.path.join(args.output_dir, args.dataset + '_community2node.json') # 'output/cora_community2node.json'
    with open(node2community_path, 'w') as f:
        f.write(json.dumps(node2community))
    with open(community2node_path, 'w') as f:
        f.write(json.dumps(community2node))
    print('finish processing community assignment, n_communities = {}'.format(n_communities))



    # calculate new features and labels using numpy
    in_feats = graph.ndata['feat'].shape[1]
    n_classes = dataset.num_classes
    print('in_feats_dim = {}, n_classes = {}'.format(in_feats, n_classes, ))
    print('calculate new features and labels...')
    X = graph.ndata['feat'].numpy()
    Y = tf.one_hot(graph.ndata['label'], depth=n_classes).numpy()# encode_onehot(graph.ndata['label'], n_classes)
    np_new_feats = np.zeros(shape=(len(community2node), in_feats), dtype='float32')
    np_new_labels = np.zeros(shape=(len(community2node), n_classes), dtype='float32')
    for community, nodes in community2node.items():
        # tf.gather() for gathering nodes in the same community
        np_new_feats[community] = np.mean(X[nodes], 0)
        np_new_labels[community] = np.mean(Y[nodes], 0)
    print('new shapes for feat and label')
    print(np_new_feats.shape, np_new_labels.shape)
    # generate new train,val,test masks
    train_mask, val_mask, test_mask = generate_masks(n_data=n_communities, test_size=args.test_split_ratio)
    # save new data for training
    np.savez_compressed(os.path.join(args.output_dir, args.dataset), 
        feat = np_new_feats, label = np_new_labels,
        train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
    # np.savez_compressed(os.path.join(args.output_dir, args.dataset + '_new_feats'), data = np_new_feats)
    # np.savez_compressed(os.path.join(args.output_dir, args.dataset + '_new_labels'), data = np_new_labels)



    print('calculate similarity...')
    sim = calculate_sim_scores(graph, node2community, community2node)
    print("{} similarity pairs are found".format(len(sim)))
    # save sim as dict
    with open(os.path.join(args.output_dir, args.dataset + '_sim.json'), 'w') as f:
        f.write(json.dumps(dict(zip([str(k) for k in sim.keys()], sim.values()))))
    # evaluate sim values
    sim_vals = list(sim.values())
    sim_vals.sort()
    # save sim_vals
    np.savez_compressed(os.path.join(args.output_dir, args.dataset + '_sim_val'), data = sim_vals)
    # plot
    try:
        fig = plt.figure()
        plt.plot(sim_vals)
        fig.suptitle('Similarity scores of {}'.format(args.dataset))#, fontsize=20)
        plt.xlabel('Index')#, fontsize=18)
        plt.ylabel('Similariy score')#, fontsize=16)
        # fig.show()
        fig.savefig('figure/' + args.dataset + '_sim_vals.png')
    except:
        print('fail to save figure')
    
    print("Preprocessing time: {:.1f} seconds".format(time.time() - preprocess_start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Connect super nodes and prepare new featurs/labels")
    parser.add_argument("--dataset", type=str, required=True,
        help= "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit")
    parser.add_argument("--test-split-ratio", type=float, default=2e-1,
                        help="ratio for valiation and test data")
    parser.add_argument("--output-dir", type=str, default="LabelPropagation/output",
                        help="source directory to retrieve super graph")
    parser.add_argument("--dgl-dir", type=str, default="/export/data/zhiyuan/.dgl",
                        help="source directory which stores DGL data")
    parser.add_argument("--ogb-dir", type=str, default="/export/data/zhiyuan/CommunityGCN/OGB/",
                        help="source directory which stores OGB data")

    args = parser.parse_args()
    print(args)

    main(args)

    my_pid = os.getpid()
    print(os.system('grep VmPeak /proc/' + str(my_pid) + '/status'))
    print(os.system('grep VmHWM /proc/' + str(my_pid) + '/status'))