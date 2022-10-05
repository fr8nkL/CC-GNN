#!/usr/bin/python
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
import sys
import csv
import os
import random
import glob
from itertools import combinations
from collections import defaultdict
import time
from tqdm import tqdm


#for comparision, this function will get the percolated k cliques in the most naive way, building all edges in the clique graph,
#before doing percolation.
def get_percolated_cliques(G, k):
    cliques = list(frozenset(c) for c in nx.find_cliques(G) if len(c) >= k)

    perc_graph = nx.Graph()
    for c1, c2 in combinations(cliques, 2):
        if len(c1.intersection(c2)) >= (k - 1):
            perc_graph.add_edge(c1, c2)

    for component in nx.connected_components(perc_graph):
        yield(frozenset.union(*component))


#this method uses the nodesToCliques dictionary, in order to only test cliques for intersection, if the cliques overlap by more than 1 node.
#Even if this improved method is used to build the full clique graph, it is still prohibitively slow, as clique graphs are often extremely large in practice.
def get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques



#This method builds only a partial set of the edges in the clique graph, as it 
def get_fast_percolated_cliques(G, k): 
    print("Starting Clique Finding. Time: ", str(datetime.time(datetime.now())))
    # print >> sys.stderr, "Starting Clique Finding. Time: ", str(datetime.time(datetime.now()))   
    cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= k]


    #randomly sampling cliques leads to a rapid falloff in NMI of results; this naive stochastic method is not appropriate.
    #randomCliques = random.sample(cliques, (len(cliques) - len(cliques)/5))
    #cliques = randomCliques

    print("Cliques found. Time: ", str(datetime.time(datetime.now())))
    # print >> sys.stderr, "Cliques found. Time: ", str(datetime.time(datetime.now()))
    nodesToCliquesDict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            nodesToCliquesDict[node].append(clique)

    print("NodesToCliques Map built. Time: ", str(datetime.time(datetime.now())))
    # print >> sys.stderr, "NodesToCliques Map built. Time: ", str(datetime.time(datetime.now()))

    cliquesToComponents = dict()
    currentComponent = 0
    
    cliquesProcessed = 0
    for clique in tqdm(cliques):
        cliquesProcessed += 1
        if cliquesProcessed % 10000 == 0:
            print("Total cliques processed: ", str(cliquesProcessed) , " time: ",  str(datetime.time(datetime.now())))
            # print >> sys.stderr, "Total cliques processed: ", str(cliquesProcessed) , " time: ",  str(datetime.time(datetime.now()))
            
        if not clique in cliquesToComponents:
            currentComponent += 1
            cliquesToComponents[clique] = currentComponent
            frontier = set()
            frontier.add(clique)
            componentCliquesProcessed = 0

            while len(frontier) > 0:
                #remove from nodes->cliques map
                #for each adjacent clique, if it percolates, add it to the frontier, and number it
                currentClique = frontier.pop()
                componentCliquesProcessed+=1
                if componentCliquesProcessed % 1000 == 0:
                    print("Component cliques processed: ", str(componentCliquesProcessed) , " time: ",  str(datetime.time(datetime.now())))
                    print("Size of frontier: ", len(frontier))
                    # print >> sys.stderr, "Component cliques processed: ", str(componentCliquesProcessed) , " time: ",  str(datetime.time(datetime.now()))
                    # print >> sys.stderr, "Size of frontier: ", len(frontier)
                    

                for neighbour in get_adjacent_cliques(currentClique, nodesToCliquesDict):
                    #this does not get appreciably faster by counting intersection size, while enumerating neighbours, in the C++ implementation this actually slightly slows performance.
                    if len(currentClique.intersection(neighbour)) >= (k-1):
                        #add to current component
                        cliquesToComponents[neighbour] = currentComponent
                        #add to the frontier
                        frontier.add(neighbour)
                        for node in neighbour:
                            nodesToCliquesDict[node].remove(neighbour)

    print("CliqueGraphComponent Built. Time: ", str(datetime.time(datetime.now()))) 
    # print >> sys.stderr, "CliqueGraphComponent Built. Time: ", str(datetime.time(datetime.now()))

    #get the union of the nodes in each component.
    #print "Number of components:" , currentComponent
    componentToNodes = defaultdict(set)
    for clique in cliquesToComponents:
        componentCliqueIn = cliquesToComponents[clique]
        componentToNodes[componentCliqueIn].update(clique)

    print("Node Components Assigned. Time: ", str(datetime.time(datetime.now()))) 
    # print >> sys.stderr, "Node Components Assigned. Time: ", str(datetime.time(datetime.now()))
    return componentToNodes.values()

def graph_reader(input_path):
    """
    Function to read graph from input path.
    :param input_path: Graph read into memory.
    :return graph: Networkx graph.
    """
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def main(args):
    print("Loading Graph. Time: ", str(datetime.time(datetime.now())))
    # print >> sys.stderr, "Loading Graph. Time: ", str(datetime.time(datetime.now()))   
    if '.csv' in args.input:
        G = graph_reader(args.input)
    else:
        # .txt file
        fh = open(args.input, "rb")
        G = nx.read_edgelist(fh)
        fh.close()
    print("Graph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    # G = nx.read_edgelist(sys.argv[1], nodetype = int, delimiter="\t")
    k = args.k # k = int(sys.argv[2])

    print("Graph Loaded. Time: ", str(datetime.time(datetime.now())))
    # print >> sys.stderr, "Graph Loaded. Time: ", str(datetime.time(datetime.now()))   
    flag = True # whether continue
    dur = []
    iter_count = 0
    while flag and iter_count < args.max_iter:
        iter_count += 1
        t0 = time.time()
        res = get_fast_percolated_cliques(G, k)
        t1 = time.time()
        print('Current k = {} | Number of communities = {} | Run time = {:.1f} seconds = {:1f} minutes |'.format(k, len(res), t1 - t0, (t1 - t0) / 60))
        if iter_count < args.max_iter:
            print('{} iterations left. Save to file? | [y] / [n]'.format(args.max_iter - iter_count))
            terminate = True if str(input()).lower().strip() == 'y' else False
            flag = not terminate
            if flag:
                print('Current k is {}. Increment k by?'.format(k))
                inc = int(input())
                k += inc
                print('k is now {}'.format(k))
        else:
            print('max iteration {} is met'.format(args.max_iter))
    # finish processing, saving to file
    print("Saving to file {}".format(args.output))
    with open(args.output, 'w') as ofs:
        for c in res:
            ofs.write(" ".join([str(x) for x in c]))
            ofs.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fast CPM")
    
    parser.add_argument("--input", type=str, required=True,
                        help=".csv input file of edge list")
    parser.add_argument("--k", type=int, default=10, 
                        help="k-clique")
    parser.add_argument("--output", type=str, default='output/community.txt', 
                        help="output file")
    parser.add_argument("--max-iter", type=int, default=1,
                        help="max limit of iterations")
    # TODO
    # parser.add_argument("--refine")
    # parser.set
    args = parser.parse_args()
    print(args)

    main(args)

    # my_pid = os.getpid()
    # print(os.system('grep VmPeak /proc/' + str(my_pid) + '/status'))
    # print(os.system('grep VmHWM /proc/' + str(my_pid) + '/status'))