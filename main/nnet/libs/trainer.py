# -*- coding: utf-8 -*-
# wujian@2018

import os
import sys
import time

from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from .utils import get_logger


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=6,
                 loss_mode="snr"):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.loss_mode = loss_mode
        print('no_impr: ', no_impr)

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        print('patience: ', patience)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "model_state_dict": self.nnet.state_dict(), 'weight'
            "optim_state_dict": self.optimizer.state_dict()
            
        }
        th.save(self.nnet.state_dict(), os.path.join(self.checkpoint,
                         "{}.pt.tar".format("best" if best else self.cur_epoch)) )
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{}.pt.tar".format("best" if best else self.cur_epoch)))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # check if save is OK
            self.save_checkpoint(best=False)
            print('start eval')
            cv = self.eval(dev_loader)
            print('end eval')
            best_loss = cv["loss"]
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0
            # make sure not inf
            self.scheduler.best = best_loss
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info("cur_lr: {}".format(cur_lr))
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                tr = self.train(train_loader)
                stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                    tr["loss"], tr["cost"], tr["batches"])
                cv = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                    cv["loss"], cv["cost"], cv["batches"])
                stats["scheduler"] = ""
                if cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv["loss"])
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, num_epochs))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def snr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        snr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        # t = th.sum(
        #     x_zm * s_zm, dim=-1,
        #     keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(s_zm) / (l2norm(s_zm - x_zm) + eps))

    def compute_loss(self, egs):
        # spks x n x S
        ests = th.nn.parallel.data_parallel(
            self.nnet, egs["mix"], device_ids=self.gpuid)
        # spks x n x S
        refs = egs["ref"]
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)
        def snr_loss(permute):
            # for snr
            return sum(
                [self.snr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # P x N
        N = egs["mix"].size(0)
        if self.loss_mode == "snr":
            sisnr_mat = th.stack(
                [snr_loss(range(num_spks))])
        elif self.loss_mode == "sisnr":
            sisnr_mat = th.stack(
                [sisnr_loss(p) for p in permutations(range(num_spks))])
        else:
            raise "must specify loss_mode to sisnr or snr"
        max_perutt, _ = th.max(sisnr_mat, dim=0)
        # si-snr
        return -th.sum(max_perutt) / N
    
    # class SingleSrcMSE(_Loss):
    r"""Measure mean square error on a batch.
    Supports both tensors with and without source axis.

    Shape:
        - est_targets: :math:`(batch, ...)`.
        - targets: :math:`(batch, ...)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)`

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # singlesrc_mse / multisrc_mse support both 'pw_pt' and 'perm_avg'.
        >>> loss_func = PITLossWrapper(singlesrc_mse, pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)
    """

    def SingleSrcMSE(self, est_targets, targets):
        r"""Compute the deep clustering loss defined in [1].

    Args:
        embedding (torch.Tensor): Estimated embeddings.
            Expected shape  :math:`(batch, frequency * frame, embedding\_dim)`.
        tgt_index (torch.Tensor): Dominating source index in each TF bin.
            Expected shape: :math:`(batch, frequency, frame)`.
        binary_mask (torch.Tensor): VAD in TF plane. Bool or Float.
            See asteroid.dsp.vad.ebased_vad.

    Returns:
         `torch.Tensor`. Deep clustering loss for every batch sample.

    Examples
        >>> import torch
        >>> from asteroid.losses.cluster import deep_clustering_loss
        >>> spk_cnt = 3
        >>> embedding = torch.randn(10, 5*400, 20)
        >>> targets = torch.LongTensor(10, 400, 5).random_(0, spk_cnt)
        >>> loss = deep_clustering_loss(embedding, targets)

    Reference
        [1] Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey
        "ALTERNATIVE OBJECTIVE FUNCTIONS FOR DEEP CLUSTERING"

    .. note::
        Be careful in viewing the embedding tensors. The target indices
        ``tgt_index`` are of shape :math:`(batch, freq, frames)`. Even if the embedding
        is of shape :math:`(batch, freq * frames, emb)`, the underlying view should be
        :math:`(batch, freq, frames, emb)` and not :math:`(batch, frames, freq, emb)`.
    """
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError(
                f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead"
            )
        loss = (targets - est_targets) ** 2
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)

    def deep_clustering_loss(embedding, tgt_index, binary_mask=None):
        pk_cnt = len(tgt_index.unique())

        batch, bins, frames = tgt_index.shape
        if binary_mask is None:
            binary_mask = th.ones(batch, bins * frames, 1)
            binary_mask = binary_mask.float()
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask.view(batch, bins * frames, 1)
        # If boolean mask, make it float.
            binary_mask = binary_mask.to(tgt_index.device)

    # Fill in one-hot vector for each TF bin
        tgt_embedding = th.zeros(batch, bins * frames, spk_cnt, device=tgt_index.device)
        tgt_embedding.scatter_(2, tgt_index.view(batch, bins * frames, 1), 1)

    # Compute VAD-weighted DC loss
        tgt_embedding = tgt_embedding * binary_mask
        embedding = embedding * binary_mask
        est_proj = th.einsum("ijk,ijl->ikl", embedding, embedding)
        true_proj = th.einsum("ijk,ijl->ikl", tgt_embedding, tgt_embedding)
        true_est_proj = th.einsum("ijk,ijl->ikl", embedding, tgt_embedding)
    # Equation (1) in [1]
        cost = batch_matrix_norm(est_proj) + batch_matrix_norm(true_proj)
        cost = cost - 2 * batch_matrix_norm(true_est_proj)
    # Divide by number of active bins, for each element in batch
        return cost / th.sum(binary_mask, dim=[1, 2])
    
def batch_matrix_norm(matrix, norm_order=2):
    """Normalize a matrix according to `norm_order`

    Args:
        matrix (torch.Tensor): Expected shape [batch, *]
        norm_order (int): Norm order.

    Returns:
        torch.Tensor, normed matrix of shape [batch]
    """
    keep_batch = list(range(1, matrix.ndim))
    return th.norm(matrix, p=norm_order, dim=keep_batch) ** norm_order