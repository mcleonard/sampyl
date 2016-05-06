from .core import np

def cat_chains(chains):
    """ Concatenate chains """
    return np.concatenate(chains).view(np.recarray)