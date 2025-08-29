import einx
import torch
from torch import nn


class C(nn.Module):
    """
    Sensitivity forward operator to do SENSE expansion
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, csm):
        return x * csm


class C_adj(nn.Module):
    """
    Sensitivity adjoint operator to do SENSE expansion
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, csm):
        return einx.sum("b [ch] x y", x * csm.conj())


class F(nn.Module):
    """
    Fourier forward operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.fft2(x, norm="ortho")


class F_adj(nn.Module):
    """
    Fourier adjoint operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.ifft2(x, norm="ortho")


class M(nn.Module):
    """
    Masking forward operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return x * mask


class MaskedForwardModel(nn.Module):
    """
    MR forward model to do SENS expansion and Fourier transform
    """

    def __init__(self):
        super().__init__()
        self.C = C()
        self.F = F()
        self.M = M()

    def forward(self, x, csm, mask):
        x = self.C(x, csm)
        x = self.F(x)
        x = self.M(x, mask)
        return x


class ForwardModel(nn.Module):
    """
    MR forward model to do SENS expansion and Fourier transform
    """

    def __init__(self):
        super().__init__()
        self.C = C()
        self.F = F()

    def forward(self, x, csm, mask):
        x = self.C(x, csm)
        x = self.F(x)
        return x
