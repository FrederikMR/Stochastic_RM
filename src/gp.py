#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:08:30 2022

@author: frederik
"""

#%% Sources

#http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf

#%% Modules

import jax.numpy as jnp
from jax import grad, jacfwd, lax, vmap, jit

import numpy as np

import kernels as km

#%% GP class

class gp_features(object):
    
    def __init__(self):
        
        self.G = None
        self.J = None

#%% Functions


def gp(X_training, y_training, sigman = 1.0, m_fun = None, k = None, Dk_fun = None, DDk_fun = None,
       max_iter = 100, tau_step = 0.1, theta_init = None, optimize=False, grad_K = None):
    
    def sim_prior(X, n_sim=10):
        
        if X.ndim == 1:
            X = X.reshape(1,-1)
        
        _, N_data = X.shape
        
        if m_fun is None:
            mu = jnp.zeros(N_data)
        else:
            mu = m_fun(X)
        
        K = km.km(X.T, kernel_fun = k_fun)
        gp = jnp.asarray(np.random.multivariate_normal(mean = mu,
                                                      cov = K,
                                                      size=n_sim))
        
        return gp
    
    def post_mom(X_test):
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)
        
        _, N_test = X_test.shape
        
        if m_fun is None:
            m_test = jnp.zeros(N_test)
        else:
            m_test = m_fun(X_test)
        
        K12 = km.km(X_training.T, X_test.T, k_fun)
        K22 = km.km(X_test.T, X_test.T, k_fun)
        
        solved = jnp.linalg.solve(K11, K12).T
        
        mu_post = m_test+solved @ (y_training-m_training)
        cov_post = K22-solved @ K12
        
        return mu_post, cov_post
    
    def sim_post(X_test, n_sim=10):
        
        mu_post, cov_post = GP.post_mom(X_test)
        gp = jnp.asarray(np.random.multivariate_normal(mean = mu_post,
                                                      cov = cov_post,
                                                      size=n_sim))
        
        return gp
    
    def ml():
        
        N_training, _ = X_training.shape
        
        sigman2 = sigman**2
        
        K11 = km.km(X_training, X_training, k_fun)+sigman2*jnp.eye(N_training)
        
        pYX = -0.5*(y_training.dot(jnp.linalg.solve(K11, y_training))+jnp.log(jnp.linalg.det(K11))-N_training*jnp.log(2*jnp.pi))
        
        return pYX
    
    def optimize_hyper(theta_init):
        
        def grad_pYX(theta):
            
            K = K11_theta(theta)
            K_inv = jnp.linalg.inv(K)
            K_theta = grad_K(theta)
            
            alpha = jnp.linalg.solve(K, y_training).reshape(-1,1)
            alpha_mat = alpha.dot(alpha.T)
                                    
            return jnp.trace((alpha_mat-K_inv).dot(K_theta))
        
        def grad_step(thetai, idx):
            
            theta = thetai+tau_step*grad_pYX(thetai)
            
            return theta, theta
            
        theta, _ = lax.scan(grad_step, init=theta_init, xs=jnp.arange(0,max_iter,1))

        return theta
    
    def jacobian_mom(X_test):

        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)
        
        n_dim, N_training = X_training.shape
        
        if m_fun is None:
            m_test = jnp.zeros(dim)
        else:
            m_test = m_fun(X_test)
            
        DK = vmap(lambda x: Dk_fun(x, X_test.reshape(-1)))(X_training.T)
        DKK = DDk_fun(X_test.reshape(-1), X_test.reshape(-1))
        
        solved = jnp.linalg.solve(K11, DK).T
        
        mu_post = m_test+solved @ (y_training-m_training)
        cov_post = DKK-solved @ DK
        
        return mu_post, cov_post
    
    def sim_jacobian(X_test):
        
        mu_post, cov_post = jacobian_mom(X_test)
        
        J = jnp.asarray(np.random.multivariate_normal(mean = mu_post,
                                                      cov = cov_post))
        
        return J
    
    def Emmf(X_test):
        
        p = 1 #y_training.shape[-1]
        mu_post, cov_post = GP.jac_mom(X_test)
        
        mu_post = mu_post.reshape(1,-1)
        
        EG = mu_post.T.dot(mu_post)+p*cov_post
        
        return EG

    def sim_mmf(X_test):
        
        J = GP.J(X_test)
        
        return J.T.dot(J)
    
    sigman2 = sigman**2
    
    if X_training.ndim == 1:
        X_training = X_training.reshape(1,-1)
    
    dim, N_training = X_training.shape
    
    if m_fun is None:
        m_training = jnp.zeros(N_training)
    else:
        m_training = m_fun(X_training)
    if k is None:
        k = km.gaussian_kernel
        
    K11_theta = lambda theta: km.km(X_training, X_training, lambda x,y: k(x,y,*theta))+sigman2*jnp.eye(N_training)
    if grad_K is None:
        grad_K = jacfwd(K11_theta)
    
    if optimize:
        theta = optimize_hyper(theta_init)
        k_fun = lambda x,y: k(x,y,*theta)
    else:
        k_fun = k
    
    if Dk_fun is None:
        Dk_fun = grad(k_fun, argnums=0)
    if DDk_fun is None:
        DDk_fun = jacfwd(grad(k_fun, argnums=0), argnums=0)
    
    K11 = km.km(X_training.T, X_training.T, k_fun)+sigman2*jnp.eye(N_training)
    
    GP = gp_features() 
    GP.sim_prior = sim_prior
    GP.post_mom = jit(post_mom)
    GP.sim_post = sim_post
    GP.ml = jit(ml)
    GP.opt = jit(optimize_hyper)
    GP.jac_mom = jit(jacobian_mom)
    GP.J = sim_jacobian
    GP.Emmf = jit(Emmf)
    GP.sim_mmf = sim_mmf
    
    return GP