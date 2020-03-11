import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import copy
from torch.utils.tensorboard import SummaryWriter

def train(model, loss_func, optimizer, num_epochs, dataloaders, dataset_sizes, device):
    # tensorboard
    writer = SummaryWriter('log/graph_classifier_experiment')

    since = time.time()
    print(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for iter, (bg, label) in enumerate(dataloaders[phase]):
                bg = bg.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(bg)
                    loss = loss_func(output, label)
                    _, preds = torch.max(output, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.detach().item()
                running_corrects += torch.sum(preds == label)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                writer.add_scalar('training loss', epoch_loss, epoch)
                writer.add_scalar('training accuracy', epoch_acc, epoch)
            else:
                writer.add_scalar('valid loss', epoch_loss, epoch)
                writer.add_scalar('valid accuracy', epoch_acc, epoch)


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model

def evaluate(model, dataloader, dataset_size, device):
    model.eval()
    running_corrects = 0

    for iter, (bg, label) in enumerate(dataloader):
        bg = bg.to(device)
        label = label.to(device)
        output = model(bg)
        _, preds = torch.max(output, 1)
        running_corrects += torch.sum(preds == label)

    epoch_acc = running_corrects.double() / dataset_size

    print('Test Acc: {:.4f}'.format(epoch_acc))

    return epoch_acc




