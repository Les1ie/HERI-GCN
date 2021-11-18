import inspect
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Union, List, Tuple, Any
from pytorch_lightning.utilities import parsing



def collect_supers_init_args(cls):
    # collect __init__ parameters for all super classes
    args = set()
    while cls.__name__ != 'object':
        valid_kwargs = inspect.signature(cls.__init__).parameters
        args.update(valid_kwargs.keys())
        # print(valid_kwargs)
        cls = cls.__base__
    return args

def parse_init_args(cls, args, **kwargs):
    # reinforced `pytorch_lightning.utilities.argparse_utils.from_argparse_args`
    if isinstance(args, ArgumentParser):
        args = cls.parse_argparser(args)

    params = vars(args)

    valid_kwargs = collect_supers_init_args(cls)
    trainer_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
    trainer_kwargs.update(**kwargs)

    return cls(**trainer_kwargs)