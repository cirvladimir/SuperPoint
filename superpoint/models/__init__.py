from .base_model import BaseModel
from collections.abc import Callable


def get_model(name) -> Callable[..., BaseModel]:
    mod = __import__('superpoint.models.{}'.format(name), fromlist=[''])
    return getattr(mod, _module_to_class(name))


def _module_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))
