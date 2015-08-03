from ..core import np
from .logps import *
import sampyl as smp
import pytest


def test_1d_MAP():
    logp = normal_1D_logp
    start = {'x': 1.}
    state = smp.find_MAP(logp, start)
