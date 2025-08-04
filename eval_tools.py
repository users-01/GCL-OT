""" Borrowed from https://arxiv.org/pdf/2307.16026.pdf"""

import torch.nn as nn
from torch.optim import Adam


import torch

class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


def LRE(x, y, idx_train, idx_val, idx_test=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = x.detach().to(device)
    input_dim = x.size()[1]
    y = y.detach().to(device)
    num_classes = y.max().item() + 1
    classifier = LogisticRegression(input_dim, num_classes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    output_fn = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()

    best_train_acc = 0
    best_val_acc = 0
    if idx_test is not None:
        best_test_acc = 0

    num_epochs = 500
    test_interval = 20


    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(x[idx_train])
        loss = criterion(output_fn(output), y[idx_train])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % test_interval == 0:
            classifier.eval()
            y_pred_train = classifier(x[idx_train]).argmax(-1).detach().cpu().numpy()
            y_pred_val = classifier(x[idx_val]).argmax(-1).detach().cpu().numpy()

            train_acc = (y[idx_train].cpu().numpy() == y_pred_train).mean() * 100
            val_acc = (y[idx_val].cpu().numpy() == y_pred_val).mean() * 100

            if train_acc > best_train_acc:
                best_train_acc = train_acc
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if idx_test is not None:
                y_pred_test = classifier(x[idx_test]).argmax(-1).detach().cpu().numpy()
                test_acc = (y[idx_test].cpu().numpy() == y_pred_test).mean() * 100
                if test_acc > best_test_acc:
                    best_test_acc = test_acc

    result = {
        'train_acc': best_train_acc,
        'val_acc': best_val_acc
    }

    if idx_test is not None:
        result['test_acc'] = best_test_acc
    
    return result