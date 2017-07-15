from __future__ import absolute_import, division, print_function

import numpy as np
import theano
import theano.tensor as T


class Optimizer(object):
    def __init__(self, lr_init=1e-3):
        self.lr = theano.shared(
            np.asarray(lr_init, dtype=theano.config.floatX), borrow=True)

    def set_learning_rate(self, lr):
        self.lr.set_value(np.asarray(lr, dtype=theano.config.floatX))

    def mult_learning_rate(self, factor=0.5):
        new_lr = self.lr.get_value() * factor
        self.lr.set_value(np.asarray(new_lr, dtype=theano.config.floatX))
        print(' * change learning rate to %.2e' % (new_lr))

    def get_updates_cost(self, cost, params, scheme='nadam'):
        if scheme == 'adagrad':
            updates = self.get_updates_adagrad(cost, params)
        elif scheme == 'adadelta':
            updates = self.get_updates_adadelta(cost, params)
        elif scheme == 'rmsprop':
            updates = self.get_updates_rmsprop(cost, params)
        elif scheme == 'adam':
            updates = self.get_updates_adam(cost, params)
        elif scheme == 'nadam':
            updates = self.get_updates_nadam(cost, params)
        elif scheme == 'sgd':
            # updates = self.get_updates_sgd_momentum(cost, params)
            updates = self.get_updates_sgd_momentum(
                cost, params, grad_clip=0.01)
        else:
            raise ValueError(
                'Select the proper scheme: '
                'adagrad / adadelta / rmsprop / adam / nadam / sgd')

        return updates

    def get_updates_adagrad(self, cost, params, eps=1e-8):
        lr = self.lr
        print(' - Adagrad: lr = %.2e' % (lr.get_value(borrow=True)))

        grads = T.grad(cost, params)
        updates = []

        for p, g in zip(params, grads):
            value = p.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=p.broadcastable)
            accu_new = accu + g ** 2
            new_p = p - (lr * g / T.sqrt(accu_new + eps))

            updates.append((accu, accu_new))
            updates.append((p, new_p))

        return updates

    def get_updates_adadelta(self, cost, params, rho=0.95, eps=1e-6):
        lr = self.lr
        print(' - Adadelta: lr = %.2e' % (lr.get_value(borrow=True)))
        one = T.constant(1.)

        grads = T.grad(cost, params)
        updates = []

        for p, g in zip(params, grads):
            value = p.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=p.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(
                np.zeros(value.shape, dtype=value.dtype),
                broadcastable=p.broadcastable)

            # update accu (as in rmsprop)
            accu_new = rho * accu + (one - rho) * g ** 2
            updates.append((accu, accu_new))

            # compute parameter update, using the 'old' delta_accu
            update = (g * T.sqrt(delta_accu + eps) /
                      T.sqrt(accu_new + eps))
            new_param = p - lr * update
            updates.append((p, new_param))

            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
            updates.append((delta_accu, delta_accu_new))

        return updates

    def get_updates_rmsprop(self, cost, params, rho=0.9, eps=1e-8):
        lr = self.lr
        print(' - RMSprop: lr = %.2e' % (lr.get_value(borrow=True)))
        one = T.constant(1.)

        grads = T.grad(cost=cost, wrt=params)

        updates = []
        for p, g in zip(params, grads):
            value = p.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=p.broadcastable)
            accu_new = rho * accu + (one - rho) * g ** 2
            gradient_scaling = T.sqrt(accu_new + eps)
            g = g / gradient_scaling

            updates.append((accu, accu_new))
            updates.append((p, p - lr * g))

        return updates

    def get_updates_adam(self, cost, params,
                         beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer.

        Parameters
        ----------
            lr: float >= 0. Learning rate.
            beta1/beta2: floats, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0.

        References
        ----------
        [1] Adam - A Method for Stochastic Optimization
        [2] Lasage:
            https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
        """
        lr = self.lr
        print(' - Adam: lr = %.2e' % (lr.get_value(borrow=True)))

        one = T.constant(1.)
        self.iterations = theano.shared(
            np.asarray(0., dtype=theano.config.floatX), borrow=True)

        grads = T.grad(cost, params)
        updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1.
        lr_t = lr * (T.sqrt(one - beta2 ** t) / (one - beta1 ** t))

        for p, g in zip(params, grads):
            p_val = p.get_value(borrow=True)
            m = theano.shared(np.zeros(p_val.shape, dtype=p_val.dtype),
                              broadcastable=p.broadcastable)
            v = theano.shared(np.zeros(p_val.shape, dtype=p_val.dtype),
                              broadcastable=p.broadcastable)

            m_t = (beta1 * m) + (one - beta1) * g
            v_t = (beta2 * v) + (one - beta2) * g ** 2
            p_t = p - lr_t * m_t / (T.sqrt(v_t) + epsilon)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))

        return updates

    def get_updates_nadam(self, cost, params,
                          beta1=0.9, beta2=0.999,
                          epsilon=1e-8, schedule_decay=0.004):
        """
        Nesterov Adam.
        Keras implementation.
        Much like Adam is essentially RMSprop with momentum,
        Nadam is Adam RMSprop with Nesterov momentum.

        Parameters
        ----------
            lr: float >= 0. Learning rate.
            beta1/beta2: floats, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0.
        References
        ----------
        [1] Nadam report - http://cs229.stanford.edu/proj2015/054_report.pdf
        [2] On the importance of initialization and momentum in deep learning -
            http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
        """
        lr = self.lr
        print(' - Nesterov Adam: lr = %.2e' % (lr.get_value(borrow=True)))

        one = T.constant(1.)
        self.iterations = theano.shared(
            np.asarray(0., dtype=theano.config.floatX), borrow=True)
        self.m_schedule = theano.shared(
            np.asarray(1., dtype=theano.config.floatX), borrow=True)
        self.beta1 = theano.shared(
            np.asarray(beta1, dtype=theano.config.floatX), borrow=True)
        self.beta2 = theano.shared(
            np.asarray(beta2, dtype=theano.config.floatX), borrow=True)
        self.schedule_decay = schedule_decay

        grads = T.grad(cost, params)
        updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1.

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta1 * (
            one - 0.5 * (T.pow(0.96, t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta1 * (
            one - 0.5 * (T.pow(0.96, (t + 1.) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = (self.m_schedule * momentum_cache_t *
                           momentum_cache_t_1)
        updates.append((self.m_schedule, m_schedule_new))

        for p, g in zip(params, grads):
            p_val = p.get_value(borrow=True)
            m = theano.shared(np.zeros(p_val.shape, dtype=p_val.dtype),
                              broadcastable=p.broadcastable)
            v = theano.shared(np.zeros(p_val.shape, dtype=p_val.dtype),
                              broadcastable=p.broadcastable)

            # the following equations given in [1]
            g_prime = g / (one - m_schedule_new)
            m_t = self.beta1 * m + (one - self.beta1) * g
            m_t_prime = m_t / (one - m_schedule_next)
            v_t = self.beta2 * v + (one - self.beta2) * g ** 2
            v_t_prime = v_t / (one - T.pow(self.beta2, t))
            m_t_bar = ((one - momentum_cache_t) * g_prime +
                       momentum_cache_t_1 * m_t_prime)

            updates.append((m, m_t))
            updates.append((v, v_t))

            p_t = p - self.lr * m_t_bar / (T.sqrt(v_t_prime) + epsilon)

            updates.append((p, p_t))
        return updates

    def get_updates_sgd_momentum(self, cost, params,
                                 decay_mode=None, decay=0.,
                                 momentum=0.9, nesterov=False,
                                 grad_clip=None, constant_clip=True):
        print(' - SGD: lr = %.2e' % (self.lr.get_value(borrow=True)), end='')
        print(', decay = %.2f' % (decay), end='')
        print(', momentum = %.2f' % (momentum), end='')
        print(', nesterov =', nesterov, end='')
        print(', grad_clip =', grad_clip)

        self.grad_clip = grad_clip
        self.constant_clip = constant_clip
        self.iterations = theano.shared(
            np.asarray(0., dtype=theano.config.floatX), borrow=True)

        # lr = self.lr_float
        lr = self.lr * (1.0 / (1.0 + decay * self.iterations))
        # lr = self.lr * (decay ** T.floor(self.iterations / decay_step))

        updates = [(self.iterations, self.iterations + 1.)]

        # Get gradients and apply clipping
        if self.grad_clip is None:
            grads = T.grad(cost, params)
        else:
            assert self.grad_clip > 0
            if self.constant_clip:
                # Constant clipping using theano.gradient.grad_clip
                clip = self.grad_clip
                grads = T.grad(
                    theano.gradient.grad_clip(cost, -clip, clip),
                    params)
            else:
                # Adaptive clipping
                clip = self.grad_clip / lr
                grads_ = T.grad(cost, params)
                grads = [T.clip(g, -clip, clip) for g in grads_]

        for p, g in zip(params, grads):
            # v_prev = theano.shared(p.get_value(borrow=True) * 0.)
            p_val = p.get_value(borrow=True)
            v_prev = theano.shared(np.zeros(p_val.shape, dtype=p_val.dtype),
                                   broadcastable=p.broadcastable)
            v = momentum * v_prev - lr * g
            updates.append((v_prev, v))

            if nesterov:
                new_p = p + momentum * v - lr * g
            else:
                new_p = p + v

            updates.append((p, new_p))
        return updates
