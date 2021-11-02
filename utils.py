import math
import time

import torch
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors as Vectors


class AverageMeter(object):
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def reset(self):
        self.val = 0.  # value
        self.avg = 0.  # average
        self.sum = 0.  # sum
        self.count = 0  # count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# init for those not in word map
def init_embeddings(embeddings):
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, emb_format, word_map):
    assert emb_format in {'glove', 'word2vec'}

    vocab = set(word_map.keys())

    print("Loading embedding...")
    cnt = 0

    if emb_format == 'glove':

        with open(emb_file, 'r', encoding='utf-8') as f:
            emb_dim = len(f.readline().split(' ')) - 1

        embeddings = torch.FloatTensor(len(vocab), emb_dim)
        init_embeddings(embeddings)

        for line in open(emb_file, 'r', encoding='utf-8'):
            line = line.split(' ')
            emb_word = line[0]

            embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            if emb_word not in vocab:
                continue
            else:
                cnt += 1

            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

        print("Number of words read: ", cnt)
        print("Number of unknown words: ", len(vocab) - cnt)

        return embeddings, emb_dim

    else:

        vectors = Vectors.load_word2vec_format(emb_file, binary=True)
        print("Load successfully")
        emb_dim = 300
        embeddings = torch.FloatTensor(len(vocab), emb_dim)
        init_embeddings(embeddings)

        for emb_word in vocab:

            if emb_word in vectors.index2word:

                embedding = vectors[emb_word]
                cnt += 1
                embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

            else:
                continue

        print("Number of words read: ", cnt)
        print("Number of unknown words: ", len(vocab) - cnt)

        return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(logits, targets):
    corrects = (torch.max(logits, 1)[1].view(targets.size()).data == targets.data).sum()
    return corrects.item() * (100.0 / targets.size(0))


def adjust_learning_rate(optimizer, current_epoch):
    frac = float(current_epoch - 10) / 50
    shrink_factor = math.pow(0.5, frac)

    print("DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))
    time.sleep(0.2)


def save_checkpoint(model_name, data_name, epoch, epochs_since_improvement, model, optimizer, acc, is_best,
                    split_file=''):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_' + data_name + '_' + model_name + split_file + '.pth'
    torch.save(state, 'checkpoints/' + filename)
    if is_best:
        torch.save(state, 'checkpoints/' + 'BEST_' + filename)


def train(train_loader, model, criterion, optimizer, epoch, device, grad_clip=None):
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    epoch_iterator = tqdm(train_loader, desc="Iteration")

    for i, (sents, labels) in enumerate(epoch_iterator):
        sents = sents.to(device)
        targets = labels.to(device)

        logits = model(sents)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        accs.update(accuracy(logits, targets))
        losses.update(loss.item())

    print('\nEpoch: [{:04}] | '
          'Loss[Value(Average)]: {loss.avg:.4f} | '
          'Accuracy[Value(Average)]: {acc.avg:.4f}'.format(epoch + 1, loss=losses, acc=accs))
    time.sleep(0.2)


def validate(val_loader, model, criterion, device):
    model = model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for i, (sents, labels) in enumerate(val_loader):

            sents = sents.to(device)
            targets = labels.to(device)

            logits = model(sents)

            loss = criterion(logits, targets)

            accs.update(accuracy(logits, targets))
            losses.update(loss.item())

        print('DEV LOSS: {loss.avg:.4f} | ACCURACY: {acc.avg:.4f}\n'.format(loss=losses, acc=accs))
        time.sleep(0.2)

    return accs.avg


def testing(test_loader, model, criterion, device):
    model = model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for i, (sents, labels) in enumerate(test_loader):
            sents = sents.to(device)
            targets = labels.to(device)

            logits = model(sents)

            loss = criterion(logits, targets)

            accs.update(accuracy(logits, targets))
            losses.update(loss.item())

        print('Test LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f}'.format(loss=losses, acc=accs))

    return accs.avg
