import os

import torch
import numpy as np
from typing import Dict, List, Union, Any, Optional
import faiss
from tqdm import tqdm
import torch.nn.functional as F

def save(model, ckpt_num, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    else:
        torch.save(model.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    print('model saved!')

def fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, nb_epoch,
        device, log_interval, start_epoch=0, save_model_to='/tmp/save_model_to'):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    # Save pre-trained model
    save(model, 0, save_model_to)

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, nb_epoch):
        scheduler.step()

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval)

        log_dict = {'epoch': epoch + 1,
                    'epoch_total': nb_epoch,
                    'loss': float(train_loss),
                    }

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, nb_epoch, train_loss)
 
        print(message)
        print(log_dict)
        if (epoch) % 5 == 0:
            X, T, *_ = predict_batchwise(model, test_loader)
            nmi, recalls = evaluate_emb(X, T, [1, 2, 4, 8])
            for i in [1, 2, 4, 8]:
                print('Recall@{}:{}'.format(i, recalls[i]))
            save(model, epoch + 1, save_model_to)

def predict_batchwise(model, dataloader):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    # print(list(model.parameters())[0].device)
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc="Batch-wise prediction"):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).cpu()
                for j in J:
                    #if i == 1: print(j)
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]


@torch.no_grad()
def evaluate_emb(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = True) -> (Any, Dict[int, float]):
    """
    More efficient way to compute the recall between samples at each k. This function uses about 8GB of memory.

    """
    nmi = None
    offset = 0
    if gallery_features is None and gallery_labels is None:
        offset = 1
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    if cosine:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])

    if hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0

        max_k = max(ks)
        index_function = faiss.GpuIndexFlatIP if cosine else faiss.GpuIndexFlatL2
        index = index_function(res, g_f.shape[1], flat_config)
    else:
        max_k = max(ks)
        index_function = faiss.IndexFlatIP if cosine else faiss.IndexFlatL2
        index = index_function(g_f.shape[1])
    index.add(g_f)
    closest_indices = index.search(q_f, max_k + offset)[1]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (q_l[:, None] == g_l[indices]).any(1).mean()
    return nmi, {k: round(v * 100, 2) for k, v in recalls.items()}



def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)

        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)

        optimizer.zero_grad()
        if loss_fn.cross_entropy_flag:
            output_embedding, output_cross_entropy = model(*data)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding, output_cross_entropy)
        else:
            output_embedding = model(*data)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding)
        total_loss += blended_loss.item()
        blended_loss.backward()

        optimizer.step()

        # Print log
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data[0]), len(train_loader.dataset), 100. * batch_idx / len(train_loader))
            for name, value in losses.items():
                message += '\t{}: {:.6f}'.format(name, np.mean(value))
 
            print(message)

    total_loss /= (batch_idx + 1)
    return total_loss
