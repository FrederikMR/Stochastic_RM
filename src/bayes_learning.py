#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:21:58 2022

@author: frederik
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap

#%% Functions

def bayes_reg(X_training, y_training, X_test, sigman=1.0, Sigmap = None, basis_fun=None):
    
    if basis_fun is None:
        Phi = X_training
        phi_test = X_test
    else:
        Phi = vmap(basis_fun)(X_training).T
        phi_test = vmap(basis_fun)(X_test).T
        
    if Phi.ndim == 1:
        Phi = Phi.reshape(1,-1)
    if phi_test.ndim == 1:
        phi_test = phi_test.reshape(1,-1)

    n_dim, N_training = Phi.shape
    
    if Sigmap is None:
        Sigmap = jnp.eye(n_dim)
        
    sigman2 = sigman**2
    
    SigmaPhi = Sigmap.dot(Phi)
    
    K = Phi.T.dot(Sigmap).dot(Phi)+jnp.eye(N_training)*sigman2
    solved = jnp.linalg.solve(K.T, SigmaPhi.T).T
    
    mu_post = phi_test.T.dot(solved).dot(y_training)
    cov_post = phi_test.T.dot(Sigmap).dot(phi_test)-phi_test.T.dot(solved).dot(SigmaPhi.T).dot(phi_test)
    
    return mu_post, cov_post
    
    