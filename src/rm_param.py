#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:12:48 2022

@author: frederik
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap

#%% Parametrized manifolds

def plane(x, coef):

    x1 = x[0]
    x2 = x[1]
    
    return jnp.array([x1, x2, jnp.sum(coef*x)])

def simple_polynomial(x, coef, n=2):
    
    x1 = x[0]
    x2 = x[1]
    
    return jnp.array([x1, x2, jnp.sum(coef*(x**n))])

def polynomial_1d(x, coef, n=2):
    
    x1 = x[0]
    x2 = x[1]
    
    return jnp.array([x1, x2, jnp.sum(coef*(x**n))])

def general_polynomial(x, coef, n=2):
    
    x1 = x[0]
    x2 = x[1]
    
    val = 0.0
    for i in range(n):
        for j in range(n):
            val += coef[i+j]*(x1**i)*(x2**j)
    
    return val

def sphere(x, R):
    
    theta = x[0]
    phi = x[1]
    
    return R*jnp.array([jnp.cos(theta)*jnp.sin(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(phi)])

def basis_rec(x, coef, a=1, b=1, N_sample=10):
    
    def fun(x1, x2, k, l):
        
        return jnp.sin(k*jnp.pi*x1/a)*jnp.sin(l*jnp.pi*x2/b)
    
    coef = coef.reshape(N_sample, N_sample)
    x1 = x[0]
    x2 = x[1]
    z = 0.0
    
    l_idx = jnp.arange(0,N_sample)
    k_idx = jnp.arange(0, N_sample)
    
    fun_val = vmap(lambda k: vmap(lambda l: fun(x1,x2,k,l))(l_idx))(k_idx)
    
    z = jnp.sum(fun_val*coef)
    
    #for k in range(N_sample):
    #    for l in range(N_sample):
    #        z += coef[k,l]*fun(x1, x2, k, l)
    
    return jnp.array([x1, x2, z])