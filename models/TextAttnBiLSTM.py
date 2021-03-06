import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from config import Config
from datasets import SSTreebankDataset
from utils import adjust_learning_rate, save_checkpoint, train, validate, testing, predict_new_sample


class ModelConfig:
    opt = Config()

    output_folder = opt.output_folder
    data_name = opt.data_name
    SST_path = opt.SST_path
    emb_file = opt.emb_file
    emb_format = opt.emb_format
    min_word_freq = opt.min_word_freq
    max_len = opt.max_len

    epochs = 120
    batch_size = 32
    workers = 0
    lr = 1.5e-4
    weight_decay = 1e-5
    decay_epoch = 20
    improvement_epoch = 10
    is_Linux = False
    store_checkpoint_epoch = 0
    checkpoint = None  # 'checkpoints/BEST_checkpoint_SST-2_TextAttnBiLSTM_4.pth'
    best_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'TextAttnBiLSTM'
    class_num = 2
    embed_dropout = 0.5
    model_dropout = 0.4
    fc_dropout = 0.3
    num_layers = 2
    embed_dim = 4096
    use_embed = True
    use_gru = True
    use_attn = False
    grad_clip = 4.  # float require


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(x)  # (batch_size, max_len, hidden_size)
        x = self.attn(x).squeeze(2)  # (batch_size, max_len)
        alpha = F.softmax(x, dim=1).unsqueeze(1)  # (batch_size, 1, max_len)
        return alpha


class ModelAttnBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pretrain_embed, use_gru, embed_dropout, fc_dropout,
                 model_dropout, num_layers, class_num, use_embed, use_attn):

        super(ModelAttnBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.use_attn = use_attn

        if use_embed:
            self.embedding = nn.Embedding(vocab_size, embed_dim).from_pretrained(pretrain_embed, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_dropout = nn.Dropout(embed_dropout)

        if use_gru:
            self.bilstm = nn.GRU(embed_dim, hidden_size, num_layers, dropout=(0 if num_layers == 1 else model_dropout),
                                 bidirectional=True, batch_first=True)
        else:
            self.bilstm = nn.LSTM(embed_dim, hidden_size, num_layers, dropout=(0 if num_layers == 1 else model_dropout),
                                  bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_size, class_num)
        self.bifc = nn.Linear(hidden_size * 2, class_num)

        self.fc_dropout = nn.Dropout(fc_dropout)

        self.attn = Attn(hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)

        # y : all output, on all timestamp, it has double size of hidden size
        # _ : only the last output
        y, _ = self.bilstm(x)

        if self.use_attn:
            y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]
            alpha = self.attn(y)
            r = alpha.bmm(y).squeeze(1)
            h = torch.tanh(r)
            logits = self.fc(h)
        else:
            y = y.view(x.shape[0], x.shape[1], 2, self.hidden_size)
            y = torch.cat([y[:, -1, 0, :], y[:, 0, 1, :]], dim=-1)
            logits = self.bifc(y)

        logits = self.fc_dropout(logits)
        return logits


def train_eval(opt, split_percent):
    best_acc = 0.
    # best_loss = 100.

    # epoch
    start_epoch = 0
    epochs = opt.epochs
    epochs_since_improvement = 0

    word_map_file = opt.output_folder + opt.data_name + '_' + 'wordmap.json'
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    embed_file = opt.output_folder + opt.data_name + '_' + 'pretrain_embed.pth'
    embed_file = torch.load(embed_file)
    pretrain_embed, embed_dim = embed_file['pretrain'], embed_file['dim']

    if opt.checkpoint is None:
        if opt.use_embed is False:
            embed_dim = opt.embed_dim
        model = ModelAttnBiLSTM(vocab_size=len(word_map),
                                embed_dim=embed_dim,
                                hidden_size=embed_dim,
                                class_num=opt.class_num,
                                pretrain_embed=pretrain_embed,
                                num_layers=opt.num_layers,
                                model_dropout=opt.model_dropout,
                                fc_dropout=opt.fc_dropout,
                                embed_dropout=opt.embed_dropout,
                                use_gru=opt.use_gru,
                                use_embed=opt.use_embed,
                                use_attn=opt.use_attn)

        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay)

    else:
        checkpoint = torch.load(opt.checkpoint, map_location='cuda:0')
        start_epoch = 0
        epochs_since_improvement = 0
        best_acc = 0
        # start_epoch = checkpoint['epoch'] + 1
        # epochs_since_improvement = checkpoint['epochs_since_improvement']
        # best_acc = checkpoint['acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(opt.device)

    criterion = nn.CrossEntropyLoss().to(opt.device)

    train_loader = torch.utils.data.DataLoader(
        SSTreebankDataset(opt.data_name, opt.output_folder, 'train_' + str(split_percent)),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers if opt.is_Linux else 0,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        SSTreebankDataset(opt.data_name, opt.output_folder, 'dev_' + str(split_percent)),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers if opt.is_Linux else 0,
        pin_memory=True)

    for epoch in range(start_epoch, epochs):
        if epoch > opt.decay_epoch:
            adjust_learning_rate(optimizer, epoch)

        if epochs_since_improvement == opt.improvement_epoch:
            print('{} Epoch No Improve, Early Stop.'.format(opt.improvement_epoch))
            break

        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              device=opt.device,
              grad_clip=opt.grad_clip)

        recent_acc, recent_loss = validate(val_loader=val_loader,
                                           model=model,
                                           criterion=criterion,
                                           device=opt.device)

        is_best = False
        if epoch > opt.store_checkpoint_epoch - 1:  # epoch start from 0
            is_best = recent_acc > best_acc  # and recent_loss < best_loss
            best_acc = max(recent_acc, best_acc)
            # best_loss = min(recent_loss, best_loss)
            if not is_best:
                epochs_since_improvement += 1
                print("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

        save_name_plus = '_' + str(split_percent)

        if epoch > opt.store_checkpoint_epoch - 1:
            save_checkpoint(opt.model_name, opt.data_name, epoch, epochs_since_improvement,
                            model, optimizer, recent_acc,
                            is_best, save_name_plus)


def test(opt, split_file, predict):
    best_model = torch.load(opt.best_model, map_location='cuda:0')
    model = best_model['model']

    model = model.to(opt.device)

    criterion = nn.CrossEntropyLoss().to(opt.device)

    test_loader = torch.utils.data.DataLoader(
        SSTreebankDataset(opt.data_name, opt.output_folder, split_file),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers if opt.is_Linux else 0,
        pin_memory=True)

    testing(test_loader, model, criterion, opt.device)
    if predict:
        predict_new_sample(opt, model, opt.device)
