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

import gp
import sp
import kernels as km

#%% Hyper-parameters

sigman = 0.5
n_training = 10
n_test = 100

domain = (-2, 2)

k = km.gaussian_kernel

#%% Noise

coef_training = sp.sim_normal(0.0, sigman**2, n_training)
coef_testing = sp.sim_normal(0.0, sigman**2, n_test)

#%% Data

def basis_fun(x):
    
    return jnp.array([x, x**2])

X_training = jnp.linspace(-1, 1, 10)
y_training = jnp.sum(basis_fun(X_training), axis=0)+coef_training

n_test = 100
X_test = jnp.linspace(-2,2,n_test)
y_test = jnp.sum(basis_fun(X_test), axis=0)+coef_testing

#%% Simulate prior distirbution

GP = gp.gp(X_training, y_training, sigman, k=k)

N_sim = 10
gp_prior = GP.sim_prior(X_test, n_sim=N_sim)
plt.figure(figsize=(8, 6))
for i in range(N_sim):
    plt.plot(X_test, gp_prior[i], linestyle='-', marker='o', markersize=3)
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$y = f(x)$', fontsize=13)
plt.title((
    'Realisations from prior distribution'))
plt.xlim(domain)
plt.show()


#%% Simualte posterior distribution

N_sim = 5
mu_post, cov_post = GP.post_mom(X_test)

gp_post = GP.sim_post(X_test, n_sim=N_sim)

std = jnp.sqrt(jnp.diag(cov_post))

fig, ax = plt.subplots(1,1,figsize=(8, 6))
# Plot the distribution of the function (mean, covariance)
ax.plot(X_test, y_test, 'b--', label='$Observations$')
ax.fill_between(X_test, mu_post-2*std, mu_post+2*std, color='red', 
                 alpha=0.15, label='$2 \sigma_{2|1}$')
ax.plot(X_test, mu_post, 'r-', lw=2, label='$\mu_{2|1}$')
ax.plot(X_training, y_training, 'ko', linewidth=2, label='$(x_1, y_1)$')
ax.plot(X_test, gp_post.T)
ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$y$', fontsize=13)
ax.set_title('Distribution of posterior and prior data.')
#ax.axis([domain[0], domain[1], -3, 3])
ax.legend()