from base_dataset import BaseDataset
from collections.abc import Callable


def get_dataset(name) -> Callable[..., BaseDataset]:
    mod = __import__('superpoint.datasets.{}'.format(name), fromlist=[''])
    return getattr(mod, _module_to_class(name))


def _module_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))
