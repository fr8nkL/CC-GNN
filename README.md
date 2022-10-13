# CC-GNN

To use CC-GNN for accelerating online GNN computation, first use the community detection package provided in the CPM or LabelPropagation folder and save the results in the corresponding output folder. Then use the get_super_graph(_cpm).py script to extract the Community Contracted Graph (CC-Graph) to the prvious output folder. Finally, use the train_super_graph.py script to perform GNN training on the super graph and obtain the reconstructed node embedding.

For example, to use LPA on the cora dataset, first go to the LabelProgation folder under root. Prepare the edge csv file in the data folder and run

```
python src/label_propagation.py --input data/cora.csv --assignment-output output/cora.json --weighting unit
```

This will put the community assignment result in the file LabelPropagation/output/cora.json. Then go to the root folder and run

```
python get_super_graph.py --dataset cora --dgl-dir <directory to your DGL datasets> --ogb-dir <directory to your OGB datasets>
```

This by default will store the CC-Graph to the LabelPropagation/output folder. Finally, run

```
python train_super_graph.py --dataset cora --gpu 0
```

for GNN training. The detailed descriptions of available parameters to pass can be found in the python scripts.

The links to other baselines can be found here.

| Model|Code|
| ---------- | ---------- |
| GraphSAGE  | https://github.com/williamleif/GraphSAGE|
| FastGCN    | https://github.com/matenure/FastGCN|
| ClusterGCN | https://github.com/google-research/google-research/tree/master/cluster_gcn|
| GraphSAINT | https://github.com/GraphSAINT/GraphSAINT|
| GBP        | https://github.com/chennnM/GBP|
| GCoarsen   | https://github.com/szzhang17/Scaling-Up-Graph-Neural-Networks-Via-Graph-Coarsening|
| GCOND      | https://github.com/ChandlerBang/GCond|
