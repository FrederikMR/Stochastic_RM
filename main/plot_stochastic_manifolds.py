#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:26:05 2022

@author: frederik
"""

#%% Modules

#Adds the source library to Python Path
import sys
sys.path.insert(1, '../src/')

import jax.numpy as jnp
from jax import vmap

import matplotlib.pyplot as plt
import matplotlib as mpl

#Own modules
import rm_param as pm
import sp
import rm

#%% Choose manifold

manifold = 'basis-rectangle'
N_sim = 10
n_points = 100
n_grid = 100

if manifold == 'plane':
    grid0x1 = -2
    grid1x1 = 2
    grid0x2 = -2
    grid1x2 = 2
    n_coef = 2
    y0 = jnp.array([1.0,1.0])
    yT = jnp.array([-1.0,1.0])
    param_fun = pm.plane
elif manifold == 'simple-polynomial':
    grid0x1 = -2
    grid1x1 = 2
    grid0x2 = -2
    grid1x2 = 2
    n_coef = 2
    y0 = jnp.array([1.0,1.0])
    yT = jnp.array([-1.0,1.0])
    N_degree = 2
    param_fun = lambda x, coef: pm.simple_polynomial(x, coef, N_degree)
elif manifold == 'polynomial-1d':
    grid0x1 = -2
    grid1x1 = 2
    grid0x2 = -2
    grid1x2 = 2
    y0 = jnp.array([1.0,1.0])
    yT = jnp.array([-1.0,1.0])
    N_degree = 2
    n_coef = N_degree
    param_fun = lambda x, coef: pm.polynomial_1d(x, coef, N_degree)
elif manifold == 'general-polynomial':
    grid0x1 = -2
    grid1x1 = 2
    grid0x2 = -2
    grid1x2 = 2
    y0 = jnp.array([1.0,1.0])
    yT = jnp.array([-1.0,1.0])
    N_degree = 2
    n_coef = N_degree
    param_fun = lambda x, coef: pm.polynomial_1d(x, coef, N_degree)
elif manifold == 'sphere':
    epsilon = 1e-2 #For numerical stability in the boundary
    grid0x1 = 0.0+epsilon
    grid1x1 = 2*jnp.pi-epsilon
    grid0x2 = 0.0+epsilon
    grid1x2 = jnp.pi-epsilon
    y0 = jnp.array([0.5*jnp.pi,0.5*jnp.pi])
    yT = jnp.array([jnp.pi,0.75*jnp.pi])
    n_coef = 1
    param_fun = pm.sphere
elif manifold == 'basis-rectangle':
    epsilon = 1e-2
    a = 2
    b = 2
    grid0x1 = 0+epsilon
    grid1x1 = a-epsilon
    grid0x2 = 0+epsilon
    grid1x2 = b-epsilon
    y0 = jnp.array([0.5,0.5])
    yT = jnp.array([1.5,1.0])
    N_sample = 3
    n_coef = N_sample**2
    param_fun = lambda x, coef: pm.basis_rec(x, coef, a=a, b=b, N_sample=N_sample)

#%% Hyper-parameters

x = jnp.linspace(grid0x1,grid1x1,n_points)
y = jnp.linspace(grid0x2,grid1x2,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points)))
X = jnp.einsum('ijk->jki', X)

#%% Simulate stochastic coefficients

N_sim = 10
dist = 'normal'
if dist == 'normal':
    mu = 0.0
    sigma = 1.0
    coef = sp.sim_normal(mu, sigma, N_sim*n_coef).reshape(N_sim, n_coef)
elif dist == 'pareto':
    mu = 2 #CHANGE THIS
    
#%% Plot realisations of the stochastic manifolds

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for i in range(N_sim):
    Z = vmap(lambda y: vmap(lambda x: param_fun(x,coef[i]))(y))(X)
    ax.plot_surface(Z[:,:,0], Z[:,:,1], Z[:,:,2], color='cyan', alpha=0.1)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$f(x_{1},x_{2})$')
ax.set_title('Realisations of stochastic Riemannian manifold')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realisations'], numpoints = 1)


plt.tight_layout()
plt.show()

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
"""
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