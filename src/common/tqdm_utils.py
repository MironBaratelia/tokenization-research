"""tqdm with ETA from global average (smoothing=0) for more stable time estimates."""
from tqdm import tqdm as _tqdm


def tqdm(iterable=None, smoothing=0, **kwargs):
    """
    Same as tqdm.tqdm but with smoothing=0 by default.
    ETA is based on average speed from start, not just recent speed.
    """
    return _tqdm(iterable, smoothing=smoothing, **kwargs)
