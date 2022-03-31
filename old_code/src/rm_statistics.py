#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:22:09 2022

@author: frederik
"""

#%% Modules

import jax.numpy as jnp
from jax import lax, vmap

#%% Functions

def km_gs(RM, X, mu0 = None, tau = 1.0, n_steps = 100): #Karcher mean, gradient search

    def step(mu, idx):
        
        Log_sum = jnp.sum(vmap(lambda x: RM.Log(mu,x))(X), axis=0)
        delta_mu = tauN*Log_sum
        mu = RM.Exp(mu, delta_mu)
        
        return mu, mu
    
    N,d = X.shape #N: number of data, d: dimension of manifold
    tauN = tau/N
    
    if mu0 is None:
        mu0 = X[0]
        
    mu, _ = lax.scan(step, init=mu0, xs=jnp.zeros(n_steps))
            
    return mu

def pga(RM, X, acceptance_rate = 0.95, normalise=False, tau = 1.0, n_steps = 100):
    
    N,d = X.shape #N: number of data, d: dimension of manifold
    mu = km_gs(RM, X, tau = 1.0, n_steps = 100)
    u = vmap(lambda x: RM.Log(mu,x))(X)
    S = u.T
    
    if normalise:
        mu_norm = S.mean(axis=0, keepdims=True)
        std = S.std(axis=0, keepdims=True)
        X_norm = (S-mu_norm)/std
    else:
        X_norm = S
    
    U,S,V = jnp.linalg.svd(X_norm,full_matrices=False)
    
    rho = jnp.cumsum((S*S) / (S*S).sum(), axis=0) 
    n = 1+len(rho[rho<acceptance_rate])
 
    U = U[:,:n]
    rho = rho[:n]
    V = V[:,:n]
    # Project the centered data onto principal component space
    Z = X_norm @ V
    
    pga_component = vmap(lambda v: RM.Exp(mu, v))(V)
    pga_proj = vmap(lambda v: RM.Exp(mu, v))(Z)
    
    return rho, U, V, Z, pga_component, pga_proj

