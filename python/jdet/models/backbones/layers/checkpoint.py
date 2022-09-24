import io
import os
import os.path as osp
import pkgutil
import re
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory












class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes = {}

    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Default: None
            logger (:mod:`logging.Logger`, optional): The logger for message.
                Default: None

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        return checkpoint_loader(filename, map_location)

