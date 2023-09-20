import numpy as np
import pickle
import math
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import wasserstein_distance
from torch import linalg as LA
from random import random

# Define a class for computing and storing average values
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Define a custom dataset class
class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, reverse):
        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")
        self.seqs = []

        for seq, label in zip(seqs, labels):
            if reverse:
                sequence = list(reversed(seq))
            else:
                sequence = seq
            self.seqs.append(np.array(sequence))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]

# Define a custom collate function for DataLoader
def visit_collate_fn(batch):
    batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = max(seq_lengths)

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
        else:
            padded = batch_seq[i]

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i])

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)
    label_tensor = torch.LongTensor(sorted_labels)

    return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths), list(sorted_indices)

# Define a function for down-sampling data
def down_sampling(x, y, rate):
    data_len = len(y)
    x_down = []
    y_down = []

    for k in range(data_len):
        if y[k] == 1 or random() < rate:
            x_down.append(x[k])
            y_down.append(y[k])

    x_down = np.array(x_down)
    y_down = np.array(y_down)

    print('After down-sampling:')
    return x_down, y_down

# Define a function to read data from a file
def read_data(file_path, start=0.0, end=0.6, downsample_rate=0.0):
    fn = open(file_path, 'rb')
    D = pickle.load(fn)
    X, Y = D[0], D[1]
    Xlength = X.shape[0]

    start_point = int(Xlength * start)
    end_point = int(Xlength * end)

    x_train = X[start_point:end_point]
    y_train = Y[start_point:end_point]

    if downsample_rate != 0.0:
        x_train, y_train = down_sampling(x_train, y_train, downsample_rate)

    return x_train, y_train

# Define a function to load a dataset
def _load_dataset(data, n_examples=None):
    batch_size = 10
    test_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, collate_fn=visit_collate_fn)

    x_test, y_test = [], []
    lengths, indices = [], []

    for i, batch in enumerate(test_loader):
        x, y, length, ind = batch
        x_test.append(x)
        y_test.append(y)
        lengths.append(length)
        indices.append(ind)

        if n_examples is not None and batch_size * i >= n_examples:
            break

    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)
    lengths = [l for sub in lengths for l in sub]
    indices = [l for sub in indices for l in sub]

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]
        lengths = lengths[:n_examples]
        indices = indices[:n_examples]

    return x_test_tensor, y_test_tensor, lengths, indices

# Define a function to compute clean accuracy
def clean_accuracy(model, x, y, lengths, batch_size=100, device=None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)
            len_curr = lengths[counter * batch_size:(counter + 1) * batch_size]

            output, alpha, beta = model(x_curr, list(len_curr))
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

# Define a function to compute the Wasserstein distance
def wass_dist_2d(cont, cont_org):
    def calc_was_dist_2d_time(u, v):
        time_num = 48
        total_w_dist = 0
        for i in range(time_num):
            u_1 = u[i, :]
            v_1 = v[i, :]
            total_w_dist += wasserstein_distance(u_1, v_1)
        return total_w_dist

    def calc_was_dist_2d_attr(u, v):
        attr = 19
        total_w_dist = 0
        for i in range(attr):
            u_1 = u[:, i]
            v_1 = v[:, i]
            total_w_dist += wasserstein_distance(u_1, v_1)
        return total_w_dist

    time_2d = cont[:, :, :48]
    attr_2d = cont[:, :, 48:]

    time_2d_org = cont_org[:, :, :48]
    attr_2d_org = cont_org[:, :, 48:]

    total_w_dist = calc_was_dist_2d_time(time_2d, time_2d_org) + calc_was_dist_2d_attr(attr_2d, attr_2d_org)
    return total_w_dist

# Define a function to compute the Frobenius norm
def frob_norm(cont, cont_org):
    diff = cont - cont_org
    norm = LA.norm(diff, 'fro')
    return norm

# Define a function to compute the similarity score
def similarity_score(cont, cont_org):
    wass_dist = wass_dist_2d(cont, cont_org)
    norm = frob_norm(cont, cont_org)
    score = wass_dist + norm
    return score

# Define a function to create a custom dataloader
def create_dataloader(data_path, batch_size=10, reverse=False, start=0.0, end=0.6, downsample_rate=0.0):
    x_train, y_train = read_data(data_path, start=start, end=end, downsample_rate=downsample_rate)
    custom_dataset = VisitSequenceWithLabelDataset(x_train, y_train, reverse=reverse)
    data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=visit_collate_fn)
    return data_loader
