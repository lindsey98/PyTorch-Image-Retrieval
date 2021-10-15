# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.sampler import BatchSampler

# Load initial models
from networks import EmbeddingNetwork

# Load batch sampler and train loss
from losses import BlendedLoss, MAIN_LOSS_CHOICES

from trainer import fit, evaluate_emb, predict_batchwise
from inference import retrieve
import dataset
from dataset import sampler

def load(file_path):
    model.load_state_dict(torch.load(file_path))
    print('model loaded!')
    return model


def infer(model, queries, db):
    retrieval_results = retrieve(model, queries, db, input_size, infer_batch_size)
    return list(zip(range(len(retrieval_results)), retrieval_results.items()))


def get_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--model-to-test', type=str)
    # Hyperparameters
    args.add_argument('--epochs', type=int, default=20)
    args.add_argument('--model', type=str,
                      choices=['densenet161', 'resnet101',  'inceptionv3', 'seresnext', 'resnet50'],
                      default='resnet50')
    args.add_argument('--input-size', type=int, default=224, help='size of input image')
    args.add_argument('--feature-extracting', type=bool, default=False)
    args.add_argument('--use-pretrained', type=bool, default=True)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--scheduler', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR'])
    args.add_argument('--attention', default=False, action='store_true')
    args.add_argument('--cross-entropy', action='store_true')
    args.add_argument('--use-augmentation', action='store_true')

    args.add_argument('--dataset', type=str, default='logo2k_super100')
    args.add_argument('--num-classes', type=int, default=107, help='number of classes for batch sampler')
    args.add_argument('--embedding-dim', type=int, default=2048, help='size of embedding dimension')
    args.add_argument('--loss-type', type=str, default='n-pair', choices=MAIN_LOSS_CHOICES)
    args.add_argument('--model-save-dir', type=str, default='checkpoints/')
    args.add_argument('--IPC', type=int, default=4)
    args.add_argument('--sz_batch', type=int, default=64)
    args.add_argument('--workers', type=int, default=4)

    return args.parse_args()


if __name__ == '__main__':
    config = get_arguments()

    data_root = '/home/ruofan/PycharmProjects/ProxyNCA-/mnt/datasets/logo2ksuperclass0.01'
    # Model parameters
    model_name = config.model
    input_size = config.input_size
    embedding_dim = config.embedding_dim
    feature_extracting = config.feature_extracting
    use_pretrained = config.use_pretrained
    attention_flag = config.attention

    # Training parameters
    nb_epoch = config.epochs
    loss_type = config.loss_type
    cross_entropy_flag = config.cross_entropy
    scheduler_name = config.scheduler
    lr = config.lr

    # Mini-batch parameters
    num_classes = config.num_classes
    use_augmentation = config.use_augmentation

    infer_batch_size = 64
    log_interval = 50

    """ Model """
    model = EmbeddingNetwork(model_name=model_name,
                             embedding_dim=embedding_dim,
                             feature_extracting=feature_extracting,
                             use_pretrained=use_pretrained,
                             attention_flag=attention_flag,
                             cross_entropy_flag=cross_entropy_flag)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    """ Load data """
    trn_dataset = dataset.load(
        name=config.dataset,
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=False
        ))

    batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(trn_dataset.ys), 8,
                                                       int(config.sz_batch / 8))

    train_loader = torch.utils.data.DataLoader(
        batch_sampler=batch_sampler,
        dataset=trn_dataset,
        num_workers=config.workers,
        pin_memory=True
    )

    ev_dataset = dataset.load(
        name=config.dataset,
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=False
        ))
    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=config.sz_batch,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True
    )
    print(len(train_loader.dataset))
    print(len(dl_ev.dataset))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Gather the parameters to be optimized/updated.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extracting:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # Send the model to GPU
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    elif scheduler_name == 'MultiStepLR':
        if use_augmentation:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1)
    else:
        raise ValueError('Invalid scheduler')

    # Loss function
    loss_fn = BlendedLoss(loss_type, cross_entropy_flag)

    # Train (fine-tune) model
    # X, T, *_ = predict_batchwise(model, dl_ev)
    # nmi, recalls = evaluate_emb(X, T, [1, 2, 4, 8])
    # for i in [1, 2, 4, 8]:
    #     print('Recall@{}:{}'.format(i, recalls[i]))

    fit(train_loader, dl_ev,  model, loss_fn, optimizer, scheduler, nb_epoch,
        device=device, log_interval=log_interval, save_model_to=config.model_save_dir)

    X, T, *_ = predict_batchwise(model, dl_ev)
    nmi, recalls = evaluate_emb(X, T, [1, 2, 4, 8])
    for i in [1, 2, 4, 8]:
        print('Recall@{}:{}'.format(i, recalls[i]))


