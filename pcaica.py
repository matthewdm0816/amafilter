from sklearn.decomposition import FastICA, KernelPCA
import pretty_errors
from icecream import ic
import torch


def PCAdenoise(pc: torch.FloatTensor, n_components: int = 100, take_components: int = 20):
    r"""
    Use PCA to denoise.
    :param: pc: torch.FloatTensor
    :param: n_components, take_components: int
    """
    # x = pc.color  # [B, N, F]
    x = pc
    b, n, f = x.shape
    x = torch.einsum("ijk->kij", x)  # [F, B, N]
    res = []
    for fx in x:  # [B, N]
        transformer = KernelPCA(n_components=n_components, kernel="rbf", fit_inverse_transform=True)
        fx_transformed = transformer.fit_transform(fx.numpy())
        ic(fx_transformed.shape)  # [B, NC]
        fx_transformed = torch.from_numpy(fx_transformed)[:, :take_components]  # [B, TC]
        ic(fx_transformed.shape)
        fx_transformed = torch.cat(
            [fx_transformed, torch.zeros((b, n_components - take_components))], dim=-1
        )  # fill noise dimensions with 0
        ic(fx_transformed.shape)
        fx = transformer.inverse_transform(fx_transformed)  # [B, N]
        res.append(torch.from_numpy(fx))
    ic(len(res))
    out = torch.stack(res) # [F, B, N]
    out = torch.einsum("kij->ijk", out)  # [B, F, N]
    return out

