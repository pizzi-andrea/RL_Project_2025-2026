import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, List, Optional, Tuple, Union
from collections import deque

def force_list(
    elements: Optional[Any] = None, to_tuple: bool = False
) -> Union[List, Tuple]:
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single
    item, already a list, or a tuple.

    Args:
        elements: The inputs as a single item, a list/tuple/deque of items, or None,
            to be converted to a list/tuple. If None, returns empty list/tuple.
        to_tuple: Whether to use tuple (instead of list).

    Returns:
        The provided item in a list of size 1, or the provided items as a
        list. If `elements` is None, returns an empty list. If `to_tuple` is True,
        returns a tuple instead of a list.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return (
        ctor()
        if elements is None
        else ctor(elements)
        if type(elements) in [list, set, tuple, deque]
        else ctor([elements])
    )


def logits2discrate(logits: torch.Tensor, unimix: float = 0.01) -> torch.Tensor:
    """
    Converte i logit in una rappresentazione categorica one-hot differenziabile,
    replicando la logica del RepresentationLayer di Ray RLlib.
    """
    # 1. Calcola le probabilità tramite softmax
    probs = torch.softmax(logits, dim=-1)

    # 2. Applica l'unimix per prevenire il collasso delle probabilità
    if unimix > 0.0:
        num_classes = logits.shape[-1]
        probs = (1.0 - unimix) * probs + unimix * (1.0 / num_classes)

    # 3. Ray ricrea i logit stabili partendo dalle probabilità mixate
    mixed_logits = torch.log(probs)

    # 4. Crea la distribuzione categorica one-hot indipendente
    distribution = torch.distributions.Independent(
        torch.distributions.OneHotCategorical(logits=mixed_logits),
        reinterpreted_batch_ndims=1,
    )

    # 5. Estrai il campione (one-hot)
    sample = distribution.sample()

    # 6. Applica lo straight-through gradient estimator
    z_st = sample.detach() + probs - probs.detach()

    return z_st


def dreamerv3_normal_initializer(parameters):
    """From Danijar Hafner's DreamerV3 JAX repo.

    Used on any layer whenever the config for that layer has `winit="normal"`.

    Note: Not identical with Glorot normal. Differs in the std computation
    glorot_std = sqrt(2/(fanin+fanout))
    this_std = sqrt(1/AVG(fanin, fanout)) / [somemagicnumber=0.879...]
    """
    for param in force_list(parameters):
        if param.dim() > 1:
            fanin, fanout = _fans(param.shape)
            scale = 1.0 / np.mean([fanin, fanout])
            std = np.sqrt(scale) / 0.87962566103423978
            with torch.no_grad():
                param.normal_(0, std)
                param.clamp_(-2, 2)


def _fans(shape):
    if len(shape) == 0:
        return 1, 1
    elif len(shape) == 1:
        return shape[0], shape[0]
    elif len(shape) == 2:
        return shape
    else:
        space = int(np.prod(shape[:-2]))
        return shape[-2] * space, shape[-1] * space