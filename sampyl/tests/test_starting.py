from ..core import np
from .logps import *
import sampyl as smp
import pytest


def test_1d_MAP():
    logp, _ = normal_1D()
    start = {'x': 1.}
    state = smp.find_MAP(logp, start)
