#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 23:01:38 2022

@author: frederik
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap

import matplotlib.pyplot as plt

import bayes_learning as bl
import sp

#%% Hyper-parameters

sigman = 0.5
n_training = 10
n_test = 100

domain = (-2, 2)

#%% Noise

coef_training = sp.sim_normal(0.0, sigman**2, n_training)
coef_testing = sp.sim_normal(0.0, sigman**2, n_test)

#%% Functions

def basis_fun(x):
    
    return jnp.array([x, x**2])

#%% Example

Sigmap = jnp.eye(1)

X_training = jnp.linspace(-1, 1, 10)
y_training = jnp.sum(basis_fun(X_training), axis=0)+coef_training

n_test = 100
X_test = jnp.linspace(-2,2,n_test)
y_test = jnp.sum(basis_fun(X_test), axis=0)+coef_testing


mu_post, cov_post = bl.bayes_reg(X_training, y_training, X_test, sigman=sigman, Sigmap = Sigmap, basis_fun=None)
std = jnp.sqrt(jnp.diag(cov_post))

domain = (-6, 6)
fig, ax = plt.subplots(1,1,figsize=(8, 6))
# Plot the distribution of the function (mean, covariance)
ax.plot(X_test, y_test, 'b--', label='$Observations$')
ax.fill_between(X_test, mu_post-2*std, mu_post+2*std, color='red', 
                 alpha=0.15, label='$2 \sigma_{2|1}$')
ax.plot(X_test, mu_post, 'r-', lw=2, label='$\mu_{2|1}$')
ax.plot(X_training, y_training, 'ko', linewidth=2, label='$(x_1, y_1)$')
ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$y$', fontsize=13)
ax.set_title('Distribution of posterior and prior data.')
#ax.axis([domain[0], domain[1], -3, 3])
ax.legend()

#%% Example

Sigmap = jnp.eye(2)

X_training = jnp.linspace(-1, 1, 10)
y_training = jnp.sum(basis_fun(X_training), axis=0)+coef_training

n_test = 100
X_test = jnp.linspace(-2,2,n_test)
y_test = jnp.sum(basis_fun(X_test), axis=0)+coef_testing

mu_post, cov_post = bl.bayes_reg(X_training, y_training, X_test, sigman=sigman, Sigmap = Sigmap, basis_fun=basis_fun)
std = jnp.sqrt(jnp.diag(cov_post))

domain = (-6, 6)
fig, ax = plt.subplots(1,1,figsize=(8, 6))
# Plot the distribution of the function (mean, covariance)
ax.plot(X_test, y_test, 'b--', label='$Observations$')
ax.fill_between(X_test, mu_post-2*std, mu_post+2*std, color='red', 
                 alpha=0.15, label='$2 \sigma_{2|1}$')
ax.plot(X_test, mu_post, 'r-', lw=2, label='$\mu_{2|1}$')
ax.plot(X_training, y_training, 'ko', linewidth=2, label='$(x_1, y_1)$')
ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$y$', fontsize=13)
ax.set_title('Distribution of posterior and prior data.')
#ax.axis([domain[0], domain[1], -3, 3])
ax.legend()
