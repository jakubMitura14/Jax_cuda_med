# this loss is adapted from https://arxiv.org/pdf/2103.03230.pdf
from functools import lru_cache

import torch
from loguru import logger
from torch import Tensor, nn

from contrastyou.losses import LossClass
from contrastyou.losses.discreteMI import compute_joint_2D_with_padding_zeros


class RedundancyCriterion(nn.Module, LossClass[Tensor]):

    def __init__(self, *, eps: float = 1e-5, symmetric: bool = True, lamda: float = 1, alpha: float) -> None:
        super().__init__()
        self._eps = eps
        self.symmetric = symmetric
        self.lamda = lamda
        self.alpha = alpha

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        k = x_out.shape[1]
        p_i_j = compute_joint_2D_with_padding_zeros(x_out=x_out, x_tf_out=x_tf_out, symmetric=self.symmetric)
        p_i_j = p_i_j.view(k, k)
        self._p_i_j = p_i_j
        target = ((self.onehot_label(k=k, device=p_i_j.device) / k) * self.alpha + p_i_j * (1 - self.alpha))
        #seem that we are summing over one axis and then repeat this sums to get desired dimensions
        
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric
        constrained = (-p_i_j * (- self.lamda * torch.log(p_j + self._eps)
                                 - self.lamda * torch.log(p_i + self._eps))
                       ).sum()
        pseudo_loss = -(target * (p_i_j + self._eps).log()).sum()
        return pseudo_loss + constrained

    @lru_cache()
    def onehot_label(self, k, device):
        return torch.eye(k, device=device, dtype=torch.bool)

    def kl_criterion(self, dist: Tensor, prior: Tensor):
        return -(prior * (dist + self._eps).log() + (1 - prior) * (1 - dist + self._eps).log()).mean()

    def get_joint_matrix(self):
        if not hasattr(self, "_p_i_j"):
            raise RuntimeError()
        return self._p_i_j.detach().cpu().numpy()

    def set_ratio(self, alpha: float):
        """
        0 : entropy minimization
        1 : barlow-twin
        """
        assert 0 <= alpha <= 1, alpha
        if self.alpha != alpha:
            logger.trace(f"Setting alpha = {alpha}")
        self.alpha = alpha

def compute_joint(x_out: Tensor, x_tf_out: Tensor, symmetric=True) -> Tensor:
    r"""
    return joint probability
    :param x_out: p1, simplex
    :param x_tf_out: p2, simplex
    :param symmetric
    :return: joint probability
    """
    # produces variable that requires grad (since args require grad)
    assert simplex(x_out), f"x_out not normalized."
    assert simplex(x_tf_out), f"x_tf_out not normalized."

    bn, k = x_out.shape
    assert x_tf_out.size()[0] == bn and x_tf_out.size()[1] == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k aggregated over one batch
    if symmetric:
        p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetric
    p_i_j /= p_i_j.sum()  # normalise

    return p_i_j.contiguous()


def compute_joint_2D(x_out: Tensor, x_tf_out: Tensor, *, symmetric: bool = True, padding: int = 0):
    k = x_out.shape[1]
    x_out = x_out.swapaxes(0, 1).contiguous()
    x_tf_out = x_tf_out.swapaxes(0, 1).contiguous()
    p_i_j = F.conv2d(
        input=x_out,
        weight=x_tf_out, padding=(int(padding), int(padding))
    )
    p_i_j = p_i_j - p_i_j.min().detach() + 1e-8

    # T x T x k x k
    p_i_j = p_i_j.permute(2, 3, 0, 1)
    p_i_j /= p_i_j.sum(dim=[2, 3], keepdim=True)  # norm

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
    p_i_j /= p_i_j.sum()  # norm
    return p_i_j.contiguous()


def compute_joint_2D_with_padding_zeros(x_out: Tensor, x_tf_out: Tensor, *, symmetric: bool = True):
    k = x_out.shape[1]
    x_out = x_out.swapaxes(0, 1).reshape(k, -1)
    N = x_out.shape[1]
    x_tf_out = x_tf_out.swapaxes(0, 1).reshape(k, -1)
    p_i_j = (x_out / math.sqrt(N)) @ (x_tf_out.t() / math.sqrt(N))
    # p_i_j = p_i_j - p_i_j.min().detach() + 1e-8

    # T x T x k x k
    # p_i_j /= p_i_j.sum()

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.t()) / 2.0
    p_i_j = p_i_j.view(1, 1, k, k)
    return p_i_j.contiguous()


def patch_generator(feature_map, patch_size=(32, 32), step_size=(16, 16)):
    b, c, h, w = feature_map.shape
    hs = np.arange(0, h - patch_size[0], step_size[0])
    hs = np.append(hs, max(h - patch_size[0], 0))
    ws = np.arange(0, w - patch_size[1], step_size[1])
    ws = np.append(ws, max(w - patch_size[1], 0))
    for _h in hs:
        for _w in ws:
            yield feature_map[:, :, _h:min(_h + patch_size[0], h), _w:min(_w + patch_size[1], w)]

