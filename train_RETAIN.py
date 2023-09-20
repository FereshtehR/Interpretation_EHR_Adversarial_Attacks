import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from tqdm import trange
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import VisitSequenceWithLabelDataset, visit_collate_fn, read_data
from settings import settings
from retain import RETAIN, retain_epoch

def main():
    # Initialize settings
    initialize_settings()

    # Load datasets
    data_train, y_train, data_valid, y_valid, data_test, y_test = load_datasets()

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(data_train, y_train, data_valid, y_valid, data_test, y_test)

    # Create and initialize the RETAIN model
    model = initialize_model()

    # Set up loss function, optimizer, and scheduler
    criterion, optimizer, scheduler = setup_training(model, train_loader)

    # Train the model, evaluate on validation set, and save the best model
    best_valid_epoch, best_valid_loss, test_loss, test_auc, test_aupr, train_losses, valid_losses = train_and_evaluate(model, criterion, optimizer, scheduler, train_loader, valid_loader, test_loader)

    # Print results
    print_results(best_valid_epoch, best_valid_loss, test_loss, test_auc, test_aupr)

def initialize_settings():
    if settings.threads == -1:
        settings.threads = torch.multiprocessing.cpu_count() - 1 or 1

    if settings.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --no-cuda")

def load_datasets():
    print('===> Loading entire datasets')

    data_train, y_train = read_data(file_path=settings.data_path, start=0.0, end=0.8, downsample_rate=0.6)
    data_valid, y_valid, data_test, y_test = split_datasets(data_train, y_train)

    return data_train, y_train, data_valid, y_valid, data_test, y_test

def split_datasets(data, labels):
    Xlength = data.shape[0]
    start, end = 0.0, 0.8
    train_start_point = int(Xlength * start)
    train_end_point = int(Xlength * end)
    data_valid = data[train_end_point:]
    y_valid = labels[train_end_point:]
    data_train = data[train_start_point:train_end_point]
    y_train = labels[train_start_point:train_end_point]

    return data_valid, y_valid, data_train, y_train

def create_data_loaders(data_train, y_train, data_valid, y_valid, data_test, y_test):
    train_set = VisitSequenceWithLabelDataset(data_train, y_train, reverse=True)
    valid_set = VisitSequenceWithLabelDataset(data_valid, y_valid, reverse=True)
    test_set = VisitSequenceWithLabelDataset(data_test, y_test, reverse=True)

    train_loader = DataLoader(dataset=train_set, batch_size=settings.batch_size, shuffle=True, collate_fn=visit_collate_fn, num_workers=settings.threads)
    valid_loader = DataLoader(dataset=valid_set, batch_size=settings.eval_batch_size, shuffle=False, collate_fn=visit_collate_fn, num_workers=settings.threads)
    test_loader = DataLoader(dataset=test_set, batch_size=settings.eval_batch_size, shuffle=False, collate_fn=visit_collate_fn, num_workers=settings.threads)

    return train_loader, valid_loader, test_loader

def initialize_model():
    model = RETAIN(
        dim_input=settings.num_features,
        dim_emb=settings.dim_emb,
        dropout_emb=settings.drop_emb,
        dim_alpha=settings.dim_alpha,
        dim_beta=settings.dim_beta,
        dropout_context=settings.drop_context,
        dim_output=2
    )

    if settings.resume:
        model.load_state_dict(torch.load(settings.resume))

    if settings.cuda:
        model = model.cuda()

    return model

def setup_training(model, train_loader):
    # Calculate class weights
    weight_class0 = torch.mean(torch.FloatTensor(train_loader.dataset.labels))
    weight_class1 = 1.0 - weight_class0
    weight = torch.FloatTensor([weight_class0, weight_class1])

    # Set the loss function
    criterion = nn.CrossEntropyLoss(weight=weight)
    if settings.cuda:
        criterion = criterion.cuda()

    # Set an optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=settings.lr,
        momentum=settings.momentum,
        nesterov=False,
        weight_decay=settings.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    return criterion, optimizer, scheduler

def train_and_evaluate(model, criterion, optimizer, scheduler, train_loader, valid_loader, test_loader):
    best_valid_epoch = 0
    best_valid_loss = sys.float_info.max

    test_loss, test_auc, test_aupr = 0, 0, 0
    train_losses = []
    valid_losses = []

    for ei in trange(settings.epochs, desc="Epochs"):
        # Train
        _, _, train_loss = retain_epoch(train_loader, model, criterion=criterion, optimizer=optimizer, train=True)
        train_losses.append(train_loss)

        # Evaluate on validation set
        _, _, valid_loss = retain_epoch(valid_loader, model, criterion=criterion)
        valid_losses.append(valid_loss)

        scheduler.step(valid_loss)

        is_best = valid_loss < best_valid_loss

        if is_best:
            best_valid_epoch = ei
            best_valid_loss = valid_loss

            torch.save(model.state_dict(), os.path.join(settings.save, 'best_model_params.pth'))

            # Evaluate on the test set
            test_y_true, test_y_pred, test_loss = retain_epoch(test_loader, model, criterion=criterion)

            if settings.cuda:
                test_y_true = test_y_true.cpu()
                test_y_pred = test_y_pred.cpu()

            test_auc = roc_auc_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")
            test_aupr = average_precision_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")

            with open(os.path.join(settings.save, 'train_result.txt'), 'w') as f:
                f.write(f'Best Validation Epoch: {ei}\n')
                f.write(f'Best Validation Loss: {valid_loss}\n')
                f.write(f'Train Loss: {train_loss}\n')
                f.write(f'Test Loss: {test_loss}\n')
                f.write(f'Test AUROC: {test_auc}\n')
                f.write(f'Test AUPR: {test_aupr}\n')

    return best_valid_epoch, best_valid_loss, test_loss, test_auc, test_aupr, train_losses, valid_losses

def print_results(best_valid_epoch, best_valid_loss, test_loss, test_auc, test_aupr):
    print(f'Best Validation Epoch: {best_valid_epoch}\n')
    print(f'Best Validation Loss: {best_valid_loss}\n')
    print(f'Test Loss: {test_loss}\n')
    print(f'Test AUROC: {test_auc}\n')
    print(f'Test AUPR: {test_aupr}\n')

if __name__ == "__main__":
    main()
