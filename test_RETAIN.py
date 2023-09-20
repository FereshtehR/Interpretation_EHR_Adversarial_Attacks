import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from utils import (
    read_data,
    VisitSequenceWithLabelDataset,
    visit_collate_fn,
)
from retain import retain_epoch
from settings import settings


def main():
    # Initialize settings
    initialize_settings()

    # Load test data
    test_seqs, test_labels = load_test_data()

    # Create test data loader
    test_loader = create_test_data_loader(test_seqs, test_labels)

    # Load the saved model
    model = load_saved_model()

    # Set loss criterion
    criterion = set_loss_criterion(model)

    # Create save directory if it doesn't exist
    create_save_directory()

    # Evaluate on the test set
    test_y_true, test_y_pred, test_loss = evaluate_test_set(test_loader, model, criterion)

    # Move data to CPU if using CUDA
    test_y_true, test_y_pred = move_data_to_cpu(test_y_true, test_y_pred)

    # Calculate different metrics
    test_auc, test_aupr, accuracy, precision, recall, f1 = calculate_metrics(test_y_true, test_y_pred)

    # Save results and print
    save_and_print_results(test_loss, test_auc, test_aupr, accuracy, precision, recall, f1)


def initialize_settings():
    if settings.threads == -1:
        settings.threads = torch.multiprocessing.cpu_count() - 1 or 1

    if settings.save == '':
        settings.save = os.path.dirname(settings.seqs_path)

    if settings.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --no-cuda")

def load_test_data():
    test_seqs, test_labels = read_data(file_path=settings.data_path, start=0.6, end=1.0)
    return test_seqs, test_labels

def create_test_data_loader(test_seqs, test_labels):
    test_set = VisitSequenceWithLabelDataset(test_seqs, test_labels, reverse=True)
    test_loader = DataLoader(dataset=test_set, batch_size=settings.eval_batch_size, shuffle=False, collate_fn=visit_collate_fn)
    return test_loader

def load_saved_model():
    model_path = os.path.join(settings.save, 'best_model_params.pth')
    model = torch.load(model_path)

    if settings.cuda:
        model = model.cuda()

    return model

def set_loss_criterion(model):
    criterion = nn.CrossEntropyLoss()
    if settings.cuda:
        criterion = criterion.cuda()
    return criterion

def create_save_directory():
    if not os.path.exists(settings.save):
        os.makedirs(settings.save)

def evaluate_test_set(test_loader, model, criterion):
    test_y_true, test_y_pred, test_loss = retain_epoch(test_loader, model, criterion=criterion)
    return test_y_true, test_y_pred, test_loss

def move_data_to_cpu(test_y_true, test_y_pred):
    if settings.cuda:
        test_y_true = test_y_true.cpu()
        test_y_pred = test_y_pred.cpu()
    return test_y_true, test_y_pred

def calculate_metrics(test_y_true, test_y_pred):
    predictions = test_y_pred.numpy()[:, 1]
    y_true = test_y_true.numpy()
    pred_labels = test_y_pred.numpy().argmax(axis=-1)
    test_auc = roc_auc_score(y_true, predictions)
    test_aupr = average_precision_score(y_true, predictions)
    accuracy = accuracy_score(y_true, pred_labels)
    precision = precision_score(y_true, pred_labels)
    recall = recall_score(y_true, pred_labels)
    f1 = f1_score(y_true, pred_labels)
    return test_auc, test_aupr, accuracy, precision, recall, f1

def save_and_print_results(test_loss, test_auc, test_aupr, accuracy, precision, recall, f1):
    with open(os.path.join(settings.save, 'test_result.txt'), 'w') as f:
        f.write('Test Loss: {}\n'.format(test_loss))
        f.write('Test AUROC: {}\n'.format(test_auc))
        f.write('Test AUPR: {}\n'.format(test_aupr))

    print("Done!")
    print('Test Loss: {}\n'.format(test_loss))
    print('Test AUROC: {}\n'.format(test_auc))
    print('Test AUPR: {}\n'.format(test_aupr))
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

if __name__ == "__main__":
    main()
