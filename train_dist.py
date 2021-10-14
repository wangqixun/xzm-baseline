import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import sys
from torch.utils.data.distributed import DistributedSampler

from rich import print

if __name__ == '__main__':
    # 1) 初始化
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


import numpy as np
import time
import json
import yaml
import pandas as pd
# import cv2
# import os
# from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import sys
# from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import torch.nn.functional as F

import esm


np.random.seed(666)

print_raw = print
def print(*info):
    if local_rank == 0:
        print_raw(*info)


def f1():
    tokenizer = AutoTokenizer.from_pretrained('../AIFuture/configs/paraphrase-xlm-r-multilingual-v1')
    model = AutoModel.from_pretrained("../AIFuture/configs/paraphrase-xlm-r-multilingual-v1")

    smiles = [
        'B1(C2=C(C=C(C=C2CO1)OC3=C(C=C(C(=N3)OCCOC(C)C)C#N)Cl)C)O',
        'B(C1=CC2=CC=CC=C2O1)(O)O',
        'Brc1cnc2[nH]cc(-c3ccccc3)c2c1',
        'C1C[C@]2(C3=C(C=CC(=C3OC[C@]2(C[C@@H]1NS(=O)(=O)C4CC4)O)F)F)S(=O)(=O)C5=CC=C(C=C5)Cl'
    ]
    encoded_input = tokenizer(smiles, padding=True, truncation=True, max_length=128, return_tensors='pt')
    print(encoded_input)

    for i in range(len(smiles)):
        a = smiles[i]
        b = tokenizer.decode(encoded_input['input_ids'][i])
        print(a)
        print(b[4:4+len(a)])


def f2():
    model, alphabet = esm.pretrained.load_model_and_alphabet_local('weights/esm1_t6_43M_UR50S.pt')
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)


    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    # debug
    # for k, v in results.items():
    #     if k == 'representations':
    #         print(k, v[6].shape)
    #     else:
    #         print(k, v.shape)

    token_representations = results["representations"][6]
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))
    print(sequence_representations)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def open_json(path_json):
    with open(path_json, 'r') as load_f:
        json_data = json.load(load_f)
    return json_data


def list_write(out_name, data_list):
    with open(out_name, 'w') as f:
        for line in data_list:
            f.write(line+'\n')


def save_checkpoint(state, outname='checkpoint_latest.pth.tar', cfg=None):
    if local_rank == 0:
        best_acc = state['best_acc']
        epoch = state['epoch']
        # filename = 'checkpoint_acc_%.4f_epoch_%02d.pth.tar' % (best_acc, epoch)
        filename = outname
        # filename = 'checkpoint_best_%d.pth.tar'
        os.makedirs(cfg['out_path'], exist_ok=True)
        filename = os.path.join(cfg['out_path'], filename)
        torch.save(state, filename)

        # best_filename = os.path.join(model_dir, 'checkpoint_best_%d.pth.tar' % name_no)
        # best_filename = filename
        # shutil.copyfile(filename, best_filename)
        print('=> Save model to %s' % filename)


def my_lr(optimizer, epoch, lr_list, mode_self_training=False):
    lr_list = np.array(lr_list, dtype=float)
    lr = lr_list[epoch - 1]
    for param_group in optimizer.param_groups:
        if mode_self_training:
            print("=> Self training Epoch: %d set lr %f => %f" % (epoch, param_group['lr'], lr))
        else:
            print("=> Epoch: %d set lr %f => %f" % (epoch, param_group['lr'], lr))
        param_group['lr'] = lr


class BaseSequence(Dataset):
    def __init__(self, data, cfg, mode='train'):
        self.data = data
        self.mode = mode
        self.max_len_x1 = cfg['max_len_x1']
        self.max_len_x2 = cfg['max_len_x2']
        self.max_len_x3 = cfg['max_len_x3']
        self.mean_y = cfg['mean_y']
        self.std_y = cfg['std_y']
        self.max_y = cfg['max_y']
        self.min_y = cfg['min_y']
        self.cfg = cfg
        self.mode = mode

    def __getitem__(self, idx):
        x0, x1, x2, x3, y = self.data[idx]
        x1 = x1[:self.max_len_x1]
        x2 = x2[:self.max_len_x2]
        if self.cfg.get('aug_flip', False) and np.random.rand() < 0.5 and self.mode=='train':
            x1, x2 = x2, x1
        x3 = x3[:self.max_len_x3]
        loss_mode = self.cfg.get('loss_mode', 'mse')
        if loss_mode == 'mse':
            y = (float(y) - self.mean_y) / self.std_y
        elif loss_mode == 'bce':
            y = (float(y) - self.min_y) / (self.max_y - self.min_y)

        if self.mode == 'train' or self.mode == 'val':
            return x1, x2, x3, y
        else:
            return x0, x1, x2, x3, y


    def __len__(self):
        return len(self.data)


class MyFinalLayer(nn.Module):
    def __init__(self, nb_in, nb_out, layers=4):
        super(MyFinalLayer, self).__init__()
        self.layers = layers
        self.fc_init = nn.Sequential(
            # nn.BatchNorm1d(768+768),
            # nn.ReLU(),
            nn.Linear(nb_in, nb_out),
            # nn.Dropout(0.2),
        )
        layers = [
            nn.Sequential(
                nn.BatchNorm1d(nb_out),
                nn.LeakyReLU(),
                nn.Linear(nb_out, nb_out),
                nn.BatchNorm1d(nb_out),
                nn.LeakyReLU(),
                nn.Linear(nb_out, nb_out),
                # nn.Dropout(0.2),
            )
            for _ in range(layers)]
        self.fc_layers = nn.ModuleList(layers)

        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(nb_out),
            nn.LeakyReLU(),
            nn.Linear(nb_out, 1),
        )

    def forward(self, x):
        x = self.fc_init(x)
        for i in range(self.layers):
            x = x + self.fc_layers[i](x)
        x = self.fc_out(x)
        return x


class Transpose(nn.Module):
    def __init__(self, ):
        super(Transpose, self).__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x


class RNNLayer(nn.Module):
    def __init__(self, nb_in, nb_out, layers=4):
        super(RNNLayer, self).__init__()
        self.layers = layers
        self.fc_init = nn.Linear(nb_in, nb_out)

        rnn_layers = [
            nn.LSTM(nb_out, nb_out, batch_first=True, )
            for _ in range(layers)
        ]
        self.rnn_layers = nn.ModuleList(rnn_layers)

    def forward(self, x):
        out = self.fc_init(x)
        for i in range(self.layers):
            out, hidden = self.rnn_layers[i](out)
            # print(out.shape)
            # print(len(hidden))
            # print(hidden[0].shape, hidden[1].shape)
        return out


class CNNLayer(nn.Module):
    def __init__(self, nb_in, nb_out, k_size=5, layers=4):
        super(CNNLayer, self).__init__()
        self.layers = layers

        self.conv_init = nn.Sequential(
            # nn.BatchNorm1d(nb_in),
            # nn.ReLU(),
            nn.Conv1d(nb_in, nb_out, k_size, padding=k_size//2),
            # nn.Dropout(0.2),
        )
        cnn_layers = [nn.Sequential(
            # nn.BatchNorm1d(nb_out),
            Transpose(),
            nn.LayerNorm(nb_out),
            Transpose(),
            nn.ReLU(),
            nn.Conv1d(nb_out, nb_out, k_size, padding=k_size//2, bias=False),
            # nn.BatchNorm1d(nb_out),
            Transpose(),
            nn.LayerNorm(nb_out),
            Transpose(),
            nn.ReLU(),
            nn.Conv1d(nb_out, nb_out, k_size, padding=k_size//2, bias=False),
            # nn.Dropout(0.2),
        ) for _ in range(layers)]
        self.cnn_layers = nn.ModuleList(cnn_layers)

        self.conv_out = nn.Sequential(
            # nn.BatchNorm1d(nb_out),
            Transpose(),
            nn.LayerNorm(nb_out),
            Transpose(),
            nn.ReLU(),
            nn.Conv1d(nb_out, nb_out, k_size, padding=k_size//2, bias=False),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        x = self.conv_init(x)
        for i in range(self.layers):
            x = x + self.cnn_layers[i](x)
        x = self.conv_out(x)

        x = torch.transpose(x, 1, 2)

        return x


class MyModel(nn.Module):
    def __init__(self, model_list, share_mode):
        super(MyModel, self).__init__()
        self.share_mode = share_mode

        if share_mode == 'x1x2x3':
            self.x1_model = model_list[0]
        elif share_mode == 'x1x2_x3':
            self.x1_model = model_list[0]
            self.x2_model = model_list[1]
        elif share_mode == 'x1_x2_x3':
            self.x1_model = model_list[0]
            self.x2_model = model_list[1]
            self.x3_model = model_list[2]

        # for p in self.parameters():
        #     p.requires_grad = False

        # self.cnn_x1 = CNNLayer(nb_in=768, nb_out=256, k_size=5, layers=4)
        # self.cnn_x2 = CNNLayer(nb_in=768, nb_out=256, k_size=5, layers=4)
        # self.middle_x1 = RNNLayer(nb_in=768, nb_out=256, layers=2)
        # self.middle_x2 = RNNLayer(nb_in=768, nb_out=256, layers=2)

        self.x1_ln = nn.LayerNorm(768)
        self.x2_ln = nn.LayerNorm(768)
        self.x3_ln = nn.LayerNorm(768)

        self.final_block = MyFinalLayer(nb_in=768*3, nb_out=1024, layers=2)

    def mean_pooling_x1(self, model_output, len_seq):
        sum_embeddings = [model_output[i:i + 1, 1: len_seq[i] + 1].mean(1) for i in range(len(model_output))]
        sum_embeddings = torch.cat(sum_embeddings, dim=0)
        if torch.cuda.is_available():
            sum_embeddings = sum_embeddings.to(device)
        return sum_embeddings

    def mean_pooling_x2(self, model_output, len_seq):
        sum_embeddings = [model_output[i:i + 1, 1: len_seq[i] + 1].mean(1) for i in range(len(model_output))]
        sum_embeddings = torch.cat(sum_embeddings, dim=0)
        if torch.cuda.is_available():
            sum_embeddings = sum_embeddings.to(device)
        return sum_embeddings

    def mean_pooling_x3(self, model_output, len_seq):
        sum_embeddings = [model_output[i:i + 1, 1: len_seq[i] + 1].mean(1) for i in range(len(model_output))]
        sum_embeddings = torch.cat(sum_embeddings, dim=0)
        if torch.cuda.is_available():
            sum_embeddings = sum_embeddings.to(device)
        return sum_embeddings

    def forward(self, x1, x2, x3, x1_embedding=None, x2_embedding=None, x3_embedding=None, infer=False):
        # if x1_embedding is None:
        #     x1_output = self.x1_model(**x1, output_hidden_states=True)
        #     x1_embedding = self.x1_ln(x1_output[0])
        #     x1_embedding = self.mean_pooling_x1(x1_embedding, x1['attention_mask'])

        # if x2_embedding is None:
        #     x2_output = self.x2_model(x2[0], repr_layers=[2], return_contacts=False)
        #     x2_embedding = self.x2_ln(x2_output["representations"][2])
        #     x2_embedding = self.mean_pooling_x2(x2_embedding, [len(seq) for _, seq in x2[1]])

        if self.share_mode == 'x1x2x3':
            x1_output = self.x1_model(x1[0], repr_layers=[6], return_contacts=False)
            x1_embedding = self.x1_ln(x1_output["representations"][6])
            x1_embedding = self.mean_pooling_x1(x1_embedding, [len(seq) for _, seq in x1[1]])
            x2_output = self.x1_model(x2[0], repr_layers=[6], return_contacts=False)
            x2_embedding = self.x2_ln(x2_output["representations"][6])
            x2_embedding = self.mean_pooling_x2(x2_embedding, [len(seq) for _, seq in x2[1]])
            x3_output = self.x1_model(x3[0], repr_layers=[6], return_contacts=False)
            x3_embedding = self.x3_ln(x3_output["representations"][6])
            x3_embedding = self.mean_pooling_x3(x3_embedding, [len(seq) for _, seq in x3[1]])
        elif self.share_mode == 'x1x2_x3':
            x1_output = self.x1_model(x1[0], repr_layers=[6], return_contacts=False)
            x1_embedding = self.x1_ln(x1_output["representations"][6])
            x1_embedding = self.mean_pooling_x1(x1_embedding, [len(seq) for _, seq in x1[1]])
            x2_output = self.x1_model(x2[0], repr_layers=[6], return_contacts=False)
            x2_embedding = self.x2_ln(x2_output["representations"][6])
            x2_embedding = self.mean_pooling_x2(x2_embedding, [len(seq) for _, seq in x2[1]])
            x3_output = self.x2_model(x3[0], repr_layers=[6], return_contacts=False)
            x3_embedding = self.x3_ln(x3_output["representations"][6])
            x3_embedding = self.mean_pooling_x3(x3_embedding, [len(seq) for _, seq in x3[1]])
        elif self.share_mode == 'x1_x2_x3':
            x1_output = self.x1_model(x1[0], repr_layers=[6], return_contacts=False)
            x1_embedding = self.x1_ln(x1_output["representations"][6])
            x1_embedding = self.mean_pooling_x1(x1_embedding, [len(seq) for _, seq in x1[1]])
            x2_output = self.x2_model(x2[0], repr_layers=[6], return_contacts=False)
            x2_embedding = self.x2_ln(x2_output["representations"][6])
            x2_embedding = self.mean_pooling_x2(x2_embedding, [len(seq) for _, seq in x2[1]])
            x3_output = self.x3_model(x3[0], repr_layers=[6], return_contacts=False)
            x3_embedding = self.x3_ln(x3_output["representations"][6])
            x3_embedding = self.mean_pooling_x3(x3_embedding, [len(seq) for _, seq in x3[1]])


        out = torch.cat([x1_embedding, x2_embedding, x3_embedding], dim=-1)
        out = self.final_block(out)

        if infer:
            return out, x1_embedding, x2_embedding, x3_embedding
        else:
            return out


def train_epoch(model, criterion, optimizer, x2_batch_converter, dataloader_tra, epoch, cfg):
    data_time = AverageMeter('- data', ':4.3f')
    batch_time = AverageMeter('- batch', ':6.3f')
    losses = AverageMeter('- loss', ':.4e')
    progress = ProgressMeter(len(dataloader_tra), data_time, batch_time, losses, 
        prefix="Epoch: [%d/%d]" % (epoch, len(cfg['lr_list'])))

    model.train()

    end = time.time()
    train_loss = 0.
    for i, (x1_batch, x2_batch, x3_batch, y_batch) in enumerate(dataloader_tra):

        data_time.update(time.time() - end)
        x1_batch = [('protein%d' % ii, x) for ii, x in enumerate(x1_batch)]
        x2_batch = [('protein%d' % ii, x) for ii, x in enumerate(x2_batch)]
        x3_batch = [('protein%d' % ii, x) for ii, x in enumerate(x3_batch)]

        _, _, x1_input_batch = x2_batch_converter(x1_batch)
        _, _, x2_input_batch = x2_batch_converter(x2_batch)
        _, _, x3_input_batch = x2_batch_converter(x3_batch)

        if torch.cuda.is_available():
            x1_input_batch = x1_input_batch.to(device)
            x2_input_batch = x2_input_batch.to(device)
            x3_input_batch = x3_input_batch.to(device)
            y_batch = y_batch.to(device)

        output_batch = model([x1_input_batch, x1_batch], [x2_input_batch, x2_batch], [x3_input_batch, x3_batch])

        loss_mode = cfg.get('loss_mode', 'mse')
        if loss_mode == 'mse':
            loss_batch = criterion(output_batch.reshape([-1, ]).double(), y_batch.reshape([-1, ]).double())
        elif loss_mode == 'bce':
            output_batch = torch.sigmoid(output_batch)
            loss_batch = criterion(output_batch.reshape([-1, 1]).double(), y_batch.reshape([-1, 1]).double())

        loss_value = loss_batch.item()
        train_loss += loss_value
        losses.update(loss_value, len(x1_batch))

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg['nb_batch_print'] == 0:
            progress.print(i)

    return train_loss


def evaluate_val(model, x2_batch_converter, val_loader, metrics_best, criterion, cfg, tra=False):
    if tra:
        print('tra evaluating ...')
    else:
        print('val evaluating ...')

    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (x1_batch, x2_batch, x3_batch, y_batch) in enumerate(val_loader):
            x1_batch = [('protein%d' % ii, x) for ii, x in enumerate(x1_batch)]
            x2_batch = [('protein%d' % ii, x) for ii, x in enumerate(x2_batch)]
            x3_batch = [('protein%d' % ii, x) for ii, x in enumerate(x3_batch)]

            _, _, x1_input_batch = x2_batch_converter(x1_batch)
            _, _, x2_input_batch = x2_batch_converter(x2_batch)
            _, _, x3_input_batch = x2_batch_converter(x3_batch)

            if torch.cuda.is_available():
                x1_input_batch = x1_input_batch.to(device)
                x2_input_batch = x2_input_batch.to(device)
                x3_input_batch = x3_input_batch.to(device)
                y_batch = y_batch.to(device)

            output_batch = model([x1_input_batch, x1_batch], [x2_input_batch, x2_batch], [x3_input_batch, x3_batch])
            output_batch = output_batch.reshape([-1, ]).double()
            target_batch = y_batch.reshape([-1, ]).double()

            loss_mode = cfg.get('loss_mode', 'mse')
            if loss_mode == 'mse':
                output_batch = output_batch * cfg['std_y'] + cfg['mean_y']
                target_batch = target_batch * cfg['std_y'] + cfg['mean_y']
            elif loss_mode == 'bce':
                output_batch = torch.sigmoid(output_batch)
                output_batch = output_batch * (cfg['max_y']-cfg['min_y']) + cfg['min_y']
                target_batch = output_batch * (cfg['max_y']-cfg['min_y']) + cfg['min_y']

            total += target_batch.size()[0]
            pos = torch.sum(torch.abs(output_batch-target_batch))
            if torch.cuda.is_available():
                pos = pos.cpu()
            correct += pos.numpy()

    metrics = float(correct) / total
    if tra:
        print('- tra_: %.4f ' % metrics)
    else:
        print('- val_: %.4f    best val_ : %.4f' % (metrics, metrics_best))
    return metrics


def train(cfg_file):
    cfg = yaml.load(open(cfg_file, 'r', encoding="utf-8", ).read(), Loader=yaml.FullLoader)
    data_dir = 'data/'
    data_file = os.path.join(data_dir, 'final_dataset_train.tsv')

    init_epoch = cfg['init_epoch']
    epochs = len(cfg['lr_list'])
    metrics_best = 9999
    loss_best = 9999.
    tra_val_ratio = 0.8

    data = open(data_file).readlines()[1:]
    data = [line.strip().split('\t') for line in data]
    data = np.array(data)
    np.random.shuffle(data)

    data_tra = data[:int(tra_val_ratio*len(data))]
    data_val = data[int(tra_val_ratio*len(data)):]

    share_mode = cfg.get('share_mode', 'x1x2x3')
    if share_mode == 'x1x2x3':
        # x2_model, x2_alphabet = esm.pretrained.load_model_and_alphabet_local('weights/esm1_t6_43M_UR50S.pt')
        x2_model, x2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        x2_batch_converter = x2_alphabet.get_batch_converter()
        model = MyModel([x2_model], share_mode)
    elif share_mode == 'x1x2_x3':
        x1_model, x2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        x2_model, x2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        x2_batch_converter = x2_alphabet.get_batch_converter()
        model = MyModel([x1_model, x2_model], share_mode)
    elif share_mode == 'x1_x2_x3':
        x1_model, x2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        x2_model, x2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        x3_model, x2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        x2_batch_converter = x2_alphabet.get_batch_converter()
        model = MyModel([x1_model, x2_model, x3_model], share_mode)
    else:
        sys.exit()

    print(cfg)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    loss_mode = cfg.get('loss_mode', 'mse')
    if loss_mode == 'mse':
        criterion = nn.MSELoss()
    elif loss_mode == 'bce':
        criterion = nn.BCELoss()
    else:
        criterion = None
        sys.exit()

    dataset_tra = BaseSequence(data_tra, cfg)
    dataset_val = BaseSequence(data_val, cfg, mode='val')
    dataloader_tra = DataLoader(
        dataset_tra, batch_size=cfg['batch_size'], num_workers=cfg['nb_worker'], pin_memory=True,
        sampler=DistributedSampler(dataset_tra))
    dataloader_val = DataLoader(
        dataset_val, batch_size=cfg['batch_size'], num_workers=cfg['nb_worker'], pin_memory=True,
        sampler=DistributedSampler(dataset_val))

    try:
        checkpoint = torch.load(cfg['resume'], map_location='cpu')
        init_epoch = checkpoint['epoch'] + 1
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        optimizer.load_state_dict(checkpoint['optimizer'])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        print("=> Resume: loaded checkpoint '{}' (epoch {})".format(cfg['resume'], checkpoint['epoch']))
    except FileNotFoundError:
        print('=> Resume: no such file or directory: "%s"' % cfg['resume'])

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    for epoch in range(init_epoch, epochs + 1):
        end_epoch = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        my_lr(optimizer, epoch, cfg['lr_list'])
        train_loss = train_epoch(model, criterion, optimizer, x2_batch_converter, dataloader_tra, epoch, cfg)

        _ = evaluate_val(model, x2_batch_converter, dataloader_tra, metrics_best, criterion, cfg, tra=True)
        metrics_val = evaluate_val(model, x2_batch_converter, dataloader_val, metrics_best, criterion, cfg)

        if metrics_val <= metrics_best:
            metrics_best, loss_best = metrics_val, train_loss

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': metrics_best,
                'optimizer': optimizer.state_dict(),
            }, outname=f'checkpoint_{metrics_best:.4f}.pth.tar', cfg=cfg)

        end_epoch = time.time() - end_epoch
        hou_epoch, min_epoch, sec_epoch = int(end_epoch / 3600), int(
            (end_epoch - 3600 * int(end_epoch / 3600)) / 60), int((end_epoch - 3600 * int(end_epoch / 3600)) % 60)
        need_epoch = end_epoch * (epochs - epoch)
        hou_need, min_need, sec_need = int(need_epoch / 3600), int(
            (need_epoch - 3600 * int(need_epoch / 3600)) / 60), int(
            (need_epoch - 3600 * int(need_epoch / 3600)) % 60)
        print('=> Single epoch: %02d:%02d:%02d' % (hou_epoch, min_epoch, sec_epoch),
              'eta: %02d:%02d:%02d' % (hou_need, min_need, sec_need))


if __name__ == '__main__':
    cfg_file = sys.argv[2]
    # 'train_dist.py', '--local_rank=0', 'configs/v17.yaml'
    print(cfg_file)
    train(cfg_file)

