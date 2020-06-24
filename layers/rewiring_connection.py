# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np
from scipy.special import expit
from scipy.stats import logistic

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.connection import Connection
from core.euler_trace_2d import EulerTrace2D


class RewiringConnection(Connection):
    """TODO"""

    def __init__(self, source, destination, transmitter, params={},
                 name="RewiringConnection"):
        """TODO"""
        super().__init__(source, destination, transmitter, name)

        if destination.get_post_size() == 0:
            return

        core.kernel.register_connection(self)

        self.size = (destination.num_compartments, source.size)

        self.n_syn_max = params.get("n_syn_max", 4)
        self.n_syn_start = params.get("n_syn_start", 4)
        self.w_max = params.get("w_max", 20.0)
        self.w_ini_min = params.get("w_ini_min", 1.0)
        self.w_ini_max = params.get("w_ini_max", 10.0)
        self.theta_ini = params.get("theta_ini", -10.0)
        self.theta_min = params.get("theta_min", -100.0)
        self.theta_max = self.w_max

        self.learn = params.get("learn", True)
        self.T = params.get("T", 1.0)
        self.eta = params.get("eta", 0.005)
        self.lambd = params.get("lambd", 1.0)
        self.gamma = params.get("gamma", 0.5)
        self.grad_sigmoid_clip = params.get("grad_sigmoid_clip", 1.0)
        self.A = params.get("A", 0.0)
        self.B = params.get("B", -1.0)
        self.stdp_th = params.get("stdp_th", -70.0)
        self.stdp_active = params.get("stdp_active", True)
        self.cichon_gan_rule = params.get("cichon_gan_rule", False)

        self.scale_w = params.get("scale_w", 0.01)
        self.scale_prior = params.get("scale_prior", 1.0)
        self.scale_likelihood = params.get("scale_likelihood", 4.0)
        self.scale_noise = np.sqrt(2 * self.T * self.eta)

        self.tau_pre = params.get("tau_pre", 20e-3)
        self.tau_pre_LTD = params.get("tau_pre_LTD", 40e-3)
        self.tau_post = params.get("tau_post", 20e-3)

        if self.cichon_gan_rule:
            self.tr_pre_LTP = EulerTrace2D(
                (self.destination.num_compartments, source.get_pre_size()), self.tau_pre)
            self.tr_pre_LTD = EulerTrace2D(
                (self.destination.num_compartments, source.get_pre_size()), self.tau_pre_LTD)
        self.tr_pre = self.get_pre_trace(self.tau_pre)
        self.tr_post = self.get_post_trace(self.tau_post)

        self.prior = np.zeros(self.size)
        self.likelihood = np.zeros(self.size)
        self.n_syn = np.zeros((destination.num_compartments, 1))

        self._init_weights()

    def _init_weights(self):
        """TODO"""
        self.c = np.zeros(self.size)
        self.w = np.zeros(self.size)
        self.theta = np.full(self.size, self.theta_ini, dtype=np.float)

        for row in self.theta:
            row[core.kernel.rng.choice(row.size, self.n_syn_start,
                                       replace=False)] =\
                core.kernel.rng.uniform(low=self.w_ini_min,
                                        high=self.w_ini_max,
                                        size=self.n_syn_start)

        np.maximum(0, self.theta, self.w)
        np.heaviside(self.theta, 0, self.c)

    def set_weights(self, theta):
        """TODO"""
        if theta.shape != self.size:
            core.logger.warning("Warning: Shape of the weight matrix"
                                " does not match (post size, pre size)"
                                " -- ingoring.")
            return

        self.c = np.zeros(self.size)
        self.w = np.zeros(self.size)
        self.theta = theta

        np.maximum(0, self.theta, self.w)
        np.heaviside(self.theta, 0, self.c)

    def evolve(self):
        """TODO"""
        if self.learn:
            # Compute number of active synapses.
            np.add.reduce(2 * (expit(self.scale_w * self.w) - 0.5),
                          axis=1, keepdims=True, out=self.n_syn)

            # Compute the prior.
            sigmoid = expit(self.lambd * (self.n_syn_max - self.n_syn))
            grad_sigmoid = logistic._pdf(self.scale_w * self.w)
            grad_sigmoid[self.w > self.grad_sigmoid_clip] = 0
            np.multiply(-self.lambd * self.scale_w * (1 - sigmoid),
                        grad_sigmoid, self.prior)

            # Compute the likelihood.
            if self.cichon_gan_rule:
                if self.destination.branch.branch_dynamics:
                    np.multiply(self.scale_likelihood * np.heaviside(self.destination.branch.pla, 0),
                                self.tr_pre_LTP.val,
                                self.likelihood)
                else:
                    np.multiply(self.scale_likelihood / self.destination.branch.v_thr
                                * self.destination.branch.pla,
                                self.tr_pre_LTP.val,
                                self.likelihood)
                np.add(-self.gamma * self.destination.branch.pla_on * self.tr_pre_LTD.val,
                       self.likelihood, self.likelihood)
            else:
                if self.destination.branch.branch_dynamics:
                    np.multiply(np.heaviside(self.destination.branch.pla, 0),
                                (self.tr_pre.val - self.gamma * (1 - self.tr_pre.val)),
                                self.likelihood)
                else:
                    np.multiply(self.destination.branch.pla / self.destination.branch.v_thr,
                                (self.tr_pre.val - self.gamma * (1 - self.tr_pre.val)),
                                self.likelihood)

            # Add contribution from prior and likelihood to active synapses.
            if self.cichon_gan_rule:
                np.add(self.theta, np.multiply(
                    self.eta, np.add(self.scale_prior * self.prior,
                                     self.likelihood)),
                       self.theta, where=self.c == 1)
            else:
                np.add(self.theta, np.multiply(
                    self.eta, np.add(self.scale_prior * self.prior,
                                     self.scale_likelihood * self.likelihood)),
                       self.theta, where=self.c == 1)

            # Add noise to all synapses.
            # W_t+dt - W_t is normally distributed with mean 0 and variance dt (N(0, dt)). So
            # rng.normal(loc=0, scale=np.sqrt(dt))
            np.add(self.theta, self.scale_noise * core.kernel.rng.normal(
                loc=0, scale=np.sqrt(simulation_timestep), size=self.size), self.theta)

            # Clip weights and parameters.
            np.clip(self.theta, self.theta_min, self.theta_max, self.theta)
            np.maximum(0, self.theta, self.w)
            np.heaviside(self.theta, 0, self.c)

    def propagate(self):
        """TODO"""
        self.propagate_forward()
        self.propagate_backward()

    def propagate_forward(self):
        """TODO"""
        for spike in self.source.get_spikes():
            self.target_state_vector += self.w[:, None, spike]

            if self.cichon_gan_rule:
                # Update LTP trace during dendritic spike and update LTD trace
                # if there is no dendritic spike.
                for i in np.where(self.destination.branch.pla > 0)[0]:
                    self.tr_pre_LTP.inc(i, spike)
                for i in np.where(self.destination.branch.pla == 0)[0]:
                    self.tr_pre_LTD.inc(i, spike)
                self.tr_pre_LTP.evolve()
                self.tr_pre_LTD.evolve()

            if self.stdp_active and self.learn:
                self.theta[((self.c[:, spike] == 1) &
                            (self.destination.branch.mem >= self.stdp_th).flatten()), spike] += self._on_pre()

                # Clip weights and parameters.
                np.clip(self.theta, self.theta_min, self.theta_max, self.theta)
                np.maximum(0, self.theta, self.w)
                np.heaviside(self.theta, 0, self.c)

    def propagate_backward(self):
        """TODO"""
        if self.stdp_active and self.learn:
            for spike in self.destination.get_spikes_immediate():
                np.add(self.theta, self._on_post(), self.theta,
                       where=(self.c == 1) & (self.destination.branch.mem >= self.stdp_th))

            # Clip weights and parameters.
            np.clip(self.theta, self.theta_min, self.theta_max, self.theta)
            np.maximum(0, self.theta, self.w)
            np.heaviside(self.theta, 0, self.c)

    def _on_pre(self):
        """TODO"""
        return self.eta * self.A * self.tr_post.val

    def _on_post(self):
        """TODO"""
        return self.eta * self.B * self.tr_pre.val
