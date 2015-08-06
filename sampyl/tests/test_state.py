from ..core import np
from ..state import State

state1 = State([('x', 1)])
state2 = State([('x', np.array([1, 2, 3]))])
state3 = State([('x', np.array([1,2,3])), ('y', 1)])
state4 = State([('x', np.array([2.]))])
state5 = State([('x', np.array([2,1,4])), ('y', 2)])

def test_add_states():
    new = state3 + state5
    print(new)
    assert(type(new) == State)
    assert(np.all(new['x'] == np.array([3, 3, 7])))
    assert(new['y'] == 3)

def test_add_list():
    new = state3 + [np.array([2, 3, 4]), 2]
    assert(type(new) == State)
    assert(np.all(new['x'] == np.array([3, 5, 7])))
    assert(new['y'] == 3)

def test_add_multi_array():
    new = state2 + np.array([2, 3, 4])
    assert(type(new) == State)
    assert(np.all(new['x'] == np.array([3, 5, 7])))

def test_add_single_array():
    new = state4 + np.array([1.])
    assert(type(new) == State)
    assert(len(new) == 1)
    assert(new['x'] == np.array([3.]))

def test_add_int():
    new = state1 + 1
    assert(type(new)==State)
    assert(new['x'] == 2)

def test_add_float():
    new = state1 + 1.
    assert(type(new)==State)
    assert(new['x'] == 2.)

def test_radd_int():
    new = 1 + state1
    assert(type(new)==State)
    assert(new['x'] == 2)

def test_radd_float():
    new = 1. + state1
    assert(type(new)==State)
    assert(new['x'] == 2.)

def test_mul_int():
    new = state1 * 2
    assert(type(new)==State)
    assert(new['x'] == 2)

def test_mul_float():
    new = state1 * 2.
    assert(type(new)==State)
    assert(new['x'] == 2.)

def test_rmul_int():
    new = 2 * state1
    assert(type(new)==State)
    assert(new['x'] == 2)

def test_rmul_float():
    new = 2. * state1
    assert(type(new)==State)
    assert(new['x'] == 2.)