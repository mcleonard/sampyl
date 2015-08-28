from ..core import np
from ..model import *
from .logps import *
from ..state import State


def test_init_model():
	model = init_model(linear_model_logp, grad_logp_flag=True)
	assert(isinstance(model, Model))

def test_init_single_model():
	model = init_model(poisson_with_grad, grad_logp=True)
	assert(isinstance(model, SingleModel))

def test_model_logp():
	model = init_model(linear_model_logp, grad_logp_flag=True)
	state = State([('b', np.ones(5)), ('sig', 1.)])
	logp_val = model.logp(state)
	assert(isinstance(logp_val, float))

def test_model_grad():
	model = init_model(linear_model_logp, grad_logp_flag=True)
	state = State([('b', np.ones(5)), ('sig', 1.)])
	grad_val = model.grad(state)
	print(grad_val)
	assert(len(grad_val) == 2)
	assert(isinstance(grad_val, State))
	assert(type(grad_val['b']) == np.ndarray)