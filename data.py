from dgl.data import MiniGCDataset
import dgl
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import torch
from torch.utils.data.sampler import SubsetRandomSampler

class Customized_MiniGCDataset(MiniGCDataset):
    def __init__(self, num_graphs, min_num_v, max_num_v):
        super(Customized_MiniGCDataset, self).__init__(num_graphs, min_num_v, max_num_v)
        self.feats = []
        self._gen_features()

        for i in range(self.num_graphs):
            self.graphs[i].ndata['feat'] = torch.tensor(self.feats[i])

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def _gen_features(self):
        for i in range(self.num_graphs):
            self.feats.append(self.graphs[i].in_degrees().view(-1, 1).float().tolist())

def visualize_graph(dgl_graph, label):
    networkx_graph = dgl_graph.to_networkx()
    label_list = ['cycle', 'star', 'wheel', 'lollipop', 'hypercube', 'grid', 'clique', 'circular ladder']

    fig, ax = plt.subplots()
    nx.draw(networkx_graph, ax=ax, with_labels=True)
    ax.set_title('Class: ' + label_list[label])
    plt.show()

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs, node_attrs='feat')

    return batched_graph, torch.tensor(labels)

def get_train_dataloader(dataset,
                   batch_size,
                   random_seed=1,
                   valid_size=0.2,
                   shuffle=True,
                   pin_memory=True):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collate, pin_memory=pin_memory
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        collate_fn=collate, pin_memory=pin_memory
    )

    dataloader = {'train': train_loader, 'val': valid_loader}
    dataset_sizes = {'train': len(dataset) - split, 'val': split}

    return {'dataloaders': dataloader, 'dataset_sizes': dataset_sizes}

def get_test_dataloader(dataset,
                   batch_size,
                   pin_memory=True):
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collate, pin_memory=pin_memory
    )
    dataset_size = len(dataset)

    return (test_loader, dataset_size)


def test_dataset():
    dataset = Customized_MiniGCDataset(320, 10, 20)
    print("Number of graphs : " + str(dataset.num_graphs))
    print("Max vertex: " + str(dataset.max_num_v))
    print("Min vertex : " + str(dataset.min_num_v))
    print("Number of dataset" + str(len(dataset)))

    random_num = random.randint(0, dataset.num_graphs)
    dgl_graph, label, feats = dataset.__getitem__(random_num)
    visualize_graph(dgl_graph, label)

    print("graph " + str(random_num) + "\'s node info")

    for i in range(len(feats)):
        print("node " + str(i) + " : " + str(feats[i]))
