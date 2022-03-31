#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:31:09 2022

@author: frederik
"""

#%% Sources

#https://bitbucket.org/stefansommer/jaxgeometry/src/main/src/Riemannian/metric.py

#%% Modules

import jax.numpy as jnp
from jax import jacfwd, jit, lax, vmap

import ode_integrator as oi

#%% manifold class

class riemannian_manifold(object):
    
    def __init__(self):
        
        self.G = None
        self.chris = None
        
#%% Jax functions

def rm_geometry(G = None, param_fun = None, n_steps = 100, grid = None, max_iter=100, tol=1e-05, method='euler'):
        
    def mmf(x):
        
        J = jacfwd(param_fun)(x)
        G = J.T.dot(J)
        
        return G
    
    def chris_symbols(x):
                
        G_inv = RM.G_inv(x)
        DG = RM.DG(x)
        
        chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                     +jnp.einsum('lij, lm->mji', DG, G_inv) \
                     -jnp.einsum('ijl, lm->mji', DG, G_inv))
        
        return chris
    
    def curvature_operator(x):
        
        Dchris = RM.Dchris(x)
        chris = RM.chris(x)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            -jnp.einsum('mikj->mijk', Dchris) \
            +jnp.einsum('sjk,mis->mijk', chris, chris) \
            -jnp.einsum('sik, mjs->mijk', chris, chris)
        
        return R
    
    def curvature_tensor(x):
        
        CO = RM.CO(x)
        G = RM.G(x)
        
        CT = jnp.einsum('sijk, sm -> ijkm', CO, G)
        
        return CT
    
    def sectional_curvature(x, e1, e2):
        
        CT = RM.CT(x)[0,1,1,0]
        G = RM.G(x)
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    def ivp_geodesic(x,v):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def bvp_geodesic(x,y):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def Exp(x,v):
        
        return RM.geo_ivp(x,v)[0][-1]
    
    def Log(x,y):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        dgamma = xt[:,len(x):]
        
        return dgamma[0]
    
    def pt(v0, gamma, Dgamma):
        
        def eq_pt(t:jnp.ndarray, v:jnp.ndarray)->jnp.ndarray:
            
            idx = jnp.argmin(jnp.abs(grid-t))
            gammat = gamma[idx]
            Dgammat = Dgamma[idx]
            
            chris = RM.chris(gammat)
            
            return -jnp.einsum('i,j,kij->k', v, Dgammat, chris)
        
        f_fun = jit(eq_pt)
        v = oi.ode_integrator(v0, f_fun, grid = grid, method=method)
        
        return v
    
    if param_fun == None and G == None:
        raise ValueError('Both the metric matrix function and parametrization are none type. One of them has to be passed!')
    
    if grid is None:
        grid = jnp.linspace(0.0, 1.0, n_steps)
    
    RM = riemannian_manifold()
    if G is not None:
        RM.G = G
        RM.G_inv = lambda x: jnp.linalg.inv(G(x))
        RM.DG = jacfwd(G)
    else:
        RM.G = jit(mmf)
        RM.G_inv = lambda x: jnp.linalg.inv(mmf(x))  
        RM.DG = jacfwd(mmf)
        
    RM.chris = jit(chris_symbols)
    RM.Dchris = jacfwd(chris_symbols)
    
    RM.CO = jit(curvature_operator)
    RM.CT = jit(curvature_tensor)
    RM.SC = jit(sectional_curvature)
    RM.geo_ivp = jit(ivp_geodesic)
    RM.geo_bvp = bvp_geodesic
    RM.Exp = Exp
    RM.Log = Log
    RM.pt = jit(pt)
    
    return RM

#%% RM statistics

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



