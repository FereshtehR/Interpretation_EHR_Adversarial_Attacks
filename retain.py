import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import AverageMeter

from tqdm import tqdm


class RETAIN(nn.Module):
    def __init__(self, dim_input, dim_emb=128, dropout_emb=0.6, dim_alpha=128, dim_beta=128, dropout_context=0.6, dim_output=2, batch_first=True):
        super(RETAIN, self).__init__()
        self.batch_first = batch_first

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(dim_input, dim_emb, bias=False),
            nn.Dropout(p=dropout_emb)
        )
        init.xavier_normal(self.embedding[0].weight)

        # RNN for alpha
        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

        # Linear layer for alpha
        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        # RNN for beta
        self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

        # Linear layer for beta
        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
        init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        self.beta_fc.bias.data.zero_()

        # Output layer
        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb, out_features=dim_output)
        )
        init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()

    def get_W(self):
        return self.output[1].weight

    def get_Wemb(self):
        return self.embedding[0].weight

    def forward(self, x, lengths, get_context=False):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        # Embedding
        emb = self.embedding(x)

        # Pack input for RNN
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

        # Calculate alpha
        g, _ = self.rnn_alpha(packed_input)
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

        mask = Variable(torch.FloatTensor([[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()

        e = self.alpha_fc(alpha_unpacked)

        # Masked softmax
        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        alpha = masked_softmax(e, mask)

        # Calculate beta
        h, _ = self.rnn_beta(packed_input)
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

        beta = F.tanh(self.beta_fc(beta_unpacked) * mask)

        # Calculate context
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # Output
        logit = self.output(context)

        if get_context:
            return logit, alpha, beta, context

        return logit, alpha, beta


def retain_epoch(loader, model, criterion, optimizer=None, train=False):
    if train and not optimizer:
        raise AttributeError("Optimizer should be given for training")

    if train:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Eval'

    losses = AverageMeter()
    labels = []
    outputs = []
    for bi, batch in enumerate(tqdm(loader, desc="{} batches".format(mode), leave=False)):
        inputs, targets, lengths, indices = batch

        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(targets)
        if next(model.parameters()).is_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        output, alpha, beta = model(input_var, lengths)
        loss = criterion(output, target_var)
        loss_output = loss.item()
        assert not np.isnan(loss_output), 'Model diverged with loss = NaN'

        sorted_indices, original_indices = zip(*sorted(enumerate(indices), key=lambda x: x[1], reverse=False))
        idx = torch.LongTensor(sorted_indices)

        labels.append(targets.gather(0, idx))
        outputs.append(F.softmax(output, dim=1).data.cpu().gather(0, torch.stack((idx, idx), dim=1)))

        losses.update(loss_output, inputs.size(0))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg
