#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:02:06 2022

@author: frederik
"""

#%% Sources

#https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/
#https://scipython.com/book/chapter-8-scipy/examples/visualizing-the-spherical-harmonics/
#https://shtools.github.io/SHTOOLS/real-spherical-harmonics.html

#%% Modules

import jax.numpy as jnp
from jax import lax, vmap

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.special import sph_harm

import sp
import rm

#%% Hyper-parameters

N_sim = 1

epsilon = 1e-2
n_points = 100
N_step = 5

y0 = jnp.array([0.5*jnp.pi,0.5*jnp.pi])
yT = jnp.array([jnp.pi,0.75*jnp.pi])

theta = jnp.linspace(0+epsilon, 2*jnp.pi-epsilon, n_points)
phi = jnp.linspace(0+epsilon, jnp.pi-epsilon, n_points)

X1, X2 = jnp.meshgrid(theta, phi)
X = jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points)))
#X = jnp.einsum('ijk->jki', X)

#%% Functions for function representation as basis functions

def r_fun(theta, phi, coef = None, N=10):
    
    #def inner_fun(carry, m):
    #    
    #    res = carry+coef[l,m]*sph_harm(m, l, theta, phi)
    #    
    #    return res, res
   # 
    #if coef is None:
    #    coef = jnp.ones((N,(N+1)**2))
    
    #val = 0.0
    #init = 0.0
    #for l in range(N):
    #    inner, _ = lax.scan(inner_fun, init=init, xs=jnp.arange(0,2*l+1,1))
    #    val += inner
    
    if coef is None:
        coef = jnp.ones((N,(N+1)**2))
        
    val = 0.0
    for l in range(N):
        for m in range(-l,l+1):
            val += coef[l,m+l]*sph_harm(m, l, theta, phi)
    
    return val
    

def param_fun(x, coef):
    
    theta = x[0]
    phi = x[1]
    
    R = r_fun(theta, phi, coef = coef, N=N_step).real
    
    return R*jnp.array([jnp.cos(theta)*jnp.sin(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(phi)])

#%% Coeeficinets

mu = 1.0
sigma = 1.0
coef = sp.sim_normal(mu, sigma, N_step*(N_step+1)**2).reshape(N_step, (N_step+1)**2)

#%% Plot realisations of the stochastic manifolds

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

Z = param_fun(X, coef)

for i in range(N_sim):
    mu = 1.0
    sigma = 1.0
    coef = sp.sim_normal(mu, sigma, N_step*(N_step+1)**2).reshape(N_step, (N_step+1)**2)
    Z = param_fun(X, coef)
    ax.plot_surface(Z[0], Z[1], Z[2], color='cyan', alpha=1.0)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$f(x_{1},x_{2})$')
ax.set_title('Realisations of stochastic Riemannian manifold')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realisations'], numpoints = 1)


plt.tight_layout()
plt.show()
"""
#%% Curvature

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for i in range(N_sim):
    sec = vmap(lambda y1: vmap(lambda y2: rm.rm_geometry(param_fun=lambda x: param_fun(x, coef[i])).SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))(y1))(X)
    ax.plot_surface(X[:,:,0], X[:,:,1], sec, color='cyan', alpha=0.1)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realised Curvature'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Geodesics

fig, ax = plt.subplots(1,2,figsize=(8,6))

gamma_ls = []
for i in range(N_sim):
    
    gamma, _ = rm.rm_geometry(param_fun=lambda x: param_fun(x, coef[i])).geo_bvp(y0,yT)
    gamma_ls.append(gamma)
    ax[0].plot(jnp.linspace(0,1,gamma.shape[0]), gamma[:,0], color='cyan', alpha=0.4)
    ax[1].plot(jnp.linspace(0,1,gamma.shape[0]), gamma[:,1], color='cyan', alpha=0.4)

ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')

for i in range(N_sim):
    Z = vmap(lambda x: param_fun(x, coef[i]))(gamma_ls[i])
    ax.plot(Z[:,0], Z[:,1], Z[:,2], color='red', alpha=1.0, linewidth=3.0)
    Z = vmap(lambda y: vmap(lambda x: param_fun(x,coef[i]))(y))(X)
    ax.plot_surface(Z[:,:,0], Z[:,:,1], Z[:,:,2], color='cyan', alpha=0.1)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$\gamma$')
ax.set_title('Realisations of stochastic geodesics')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
ax.legend([legend], ['Realisations'], numpoints = 1)
"""

