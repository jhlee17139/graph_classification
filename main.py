import argparse
from data import Customized_MiniGCDataset
from data import get_train_dataloader
from data import get_test_dataloader
from model import GraphClassifier
from train import train
from train import evaluate
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def main(args):
    in_feats = 1
    n_hidden = 256
    n_classes = 8
    n_layers = 1
    activation= F.relu
    dropout = 0.2
    weight_path = './weight/classifier.pth'
    num_epochs = args.n_epochs
    mode = args.mode

    # prepare dataset
    train_dataset = Customized_MiniGCDataset(2000, 10, 30)
    train_data_dict = get_train_dataloader(train_dataset, 32)
    train_dataloaders = train_data_dict['dataloaders']
    train_dataset_sizes = train_data_dict['dataset_sizes']

    test_dataset = Customized_MiniGCDataset(500, 10, 30)
    test_dataloader, test_dataset_size = get_test_dataloader(test_dataset, 32)

    # GraphClassifier
    model = GraphClassifier(in_feats, n_hidden, n_classes, n_layers, activation, dropout)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train mode
    if mode == 'train':
        model = model.to(device)
        model = train(model, loss_func, optimizer, num_epochs, train_dataloaders, train_dataset_sizes, device)
        torch.save(model.state_dict(), weight_path)

    # test mode
    else:
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        acc = evaluate(model, test_dataloader, test_dataset_size, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphClassifier')
    parser.add_argument("--n-epochs", type=int, default=700,
                        help="number of training epochs")
    parser.add_argument("--mode", default='test')
    args = parser.parse_args()
    main(args)

