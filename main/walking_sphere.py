#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 01:22:24 2022

@author: frederik
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap

from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib as mpl

import gp
import kernels as km
import rm

#%% Loading data

knee_sphere = loadmat('../data/sphere_walking.mat', squeeze_me=True)
data = jnp.asarray(knee_sphere['data'])

#%% Hyper-parameters

sigman = 1.0
k = km.gaussian_kernel

n_points = 100
grid0x1 = -1
grid1x1 = 1
grid0x2 = -1
grid1x2 = 1

x = jnp.linspace(grid0x1,grid1x1,n_points)
y = jnp.linspace(grid0x2,grid1x2,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points)))
X = jnp.einsum('ijk->jki', X)

#%% Learning metric matrix function with Gaussian process

X_training = data[0:2,:]
y_training = data[-1,:]

GP = gp.gp(X_training, y_training, sigman, k=k)

RM = rm.rm_geometry(G=GP.Emmf)

#%% Plot realisations for the learned manifold
"""
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

Z_sim = vmap(lambda y1: vmap(lambda y2: GP.sim_post(y2, n_sim=10))(y1))(X)
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

"""

#%% Geodesics

y0 = X_training[:,0]
yT = X_training[:,3]
gamma, _ = RM.geo_bvp(y0,yT)

fig, ax = plt.subplots(1,2,figsize=(8,6))
ax[0].plot(jnp.linspace(0,1,gamma.shape[0]), gamma[:,0], color='red', alpha=1.0)
ax[1].plot(jnp.linspace(0,1,gamma.shape[0]), gamma[:,1], color='red', alpha=1.0)

gamma = gamma.T
gamma_manifold, _ = GP.post_mom(gamma)

theta = jnp.linspace(0,2*jnp.pi,100)
phi = jnp.linspace(0,jnp.pi,100)
theta, phi = jnp.meshgrid(theta, phi)

sphere = jnp.array([jnp.cos(theta)*jnp.sin(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(phi)])

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
Z = jnp.concatenate((gamma, gamma_manifold.reshape(1,-1)))
ax.plot(Z[0], Z[1], Z[2], color='red', alpha=1.0, linewidth=3.0)
ax.scatter(data[0], data[1], data[2], color='black', alpha=1.0, linewidth=0.1)
ax.scatter(sphere[0], sphere[1], sphere[2], color='cyan', alpha=0.1)

#%% Curvature

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sec = vmap(lambda y1: vmap(lambda y2: RM.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))(y1))(X)
ax.plot_surface(X[:,:,0], X[:,:,1], sec, color='cyan', alpha=1.0)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.axes.set_zlim3d(bottom=-0.5, top=0.5) 
ax.set_title('Realisations of stochastic sectional curvature')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realised Curvature'], numpoints = 1)