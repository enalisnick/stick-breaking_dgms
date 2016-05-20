import numpy as np
import theano
import theano.tensor as T

from utils.utils import sharedX

### ADAM ###

def get_adam_updates(cost, params, lr=0.0003, b1=0.95, b2=0.999, e=1e-8):
    """ Adam optimizer
    """
    gradients = dict(zip(params, T.grad(cost, params)))
    return get_adam_updates_from_gradients(gradients, lr=lr, b1=b1, b2=b2, e=e)


def get_adam_updates_from_gradients(gradients, lr=0.0003, b1=0.95, b2=0.999, e=1e-8):
    """ Adam optimizer

    Parameters
    ----------
    gradients : dict
        Dict object where each entry has key: param, value: gparam.
    """
    updates = []
    i = theano.shared(np.asarray(0., dtype=theano.config.floatX))
    i_t = i + 1.
    for p, g in gradients.items():
        m = sharedX(p.get_value()*0.)
        v = sharedX(p.get_value()*0.)

        m_t = (b1 * m) + ((1. - b1) * g)
        v_t = (b2 * v) + ((1. - b2) * g**2)
        m_t_hat = m_t / (1. - b1**i_t)
        v_t_hat = v_t / (1. - b2**i_t)
        p_t = p - lr * m_t_hat / (T.sqrt(v_t_hat) + e)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates
