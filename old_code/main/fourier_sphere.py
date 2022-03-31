#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:06:38 2022

@author: frederik
"""

#%% Modules

#Adds the source library to Python Path
import sys
sys.path.insert(1, '../src/')

import numpy as np

import jax.numpy as jnp
from jax import vmap

import matplotlib.pyplot as plt
import matplotlib as mpl

from sympy.plotting import plot_implicit

import dif_geo_sympy
import dif_geo_jax

import sympy as sym


#%% Radius fun

def r_fun_sym(phi, theta, coef, R=1):
    
    n = len(coef)
    
    r = 0.0
    for i in range(n):
        r += coef[i]*sym.cos(i*theta/(2*np.pi))*sym.cos(i*phi/(np.pi))
    
    return R*r

def r_fun_np(phi, theta, coef, R=1):
    
    n = len(coef)
    
    r = 0.0
    for i in range(n):
        r += coef[i]*np.cos(i*theta/(2*np.pi))*np.cos(i*phi/(np.pi))
    
    return R*r

def f_fun(x, coef, R=1):
    
    n = len(coef)
    
    r = 0.0
    for i in range(n):
        r += coef[i]*jnp.cos(i*x[0]/(2*jnp.pi))*jnp.cos(i*x[1]/(jnp.pi))
    
    return jnp.array([R*r*jnp.cos(x[0])*jnp.sin(x[1]), R*r*jnp.sin(x[0])*jnp.sin(x[1]), R*r*jnp.cos(x[1])]).reshape(-1)

#%% Hyper-parameter

np.random.seed(2712)

sigma = 0.1
mu = 1.0
n_points = 100
grid0 = -np.pi/2
grid1 = np.pi/2
N_sim = 10
n_coef = 10

phi, theta = sym.symbols('phi theta')
x = sym.Matrix([phi, theta])
coef = sym.symbols('w0:%d'%n_coef)

param_fun = sym.Matrix([r_fun_sym(phi,theta,coef)*sym.cos(theta)*sym.sin(phi),
                        r_fun_sym(phi,theta,coef)*sym.sin(theta)*sym.sin(phi),
                        r_fun_sym(phi,theta,coef)*sym.cos(phi)])

#%% Initial Computations

#_ = dif_geo_sympy.compute_mmf(param_fun, x, print_latex = True)
#_ = dif_geo_sympy.chris_symbols(x, param_fun=param_fun, print_latex=True)
#_ = dif_geo_sympy.eq_geodesics(x, param_fun=param_fun, print_latex=True)
#_ = dif_geo_sympy.sectional_curvature2d(x, param_fun=param_fun, print_latex=True)

#%% Plot realisations as well as expected parametrization (NOT EQUAL TO EXPECTED METRIC)

coef = np.random.normal(mu, sigma, [N_sim, n_coef])
alpha = 5.0
xm = 1.0
coef = (np.random.pareto(alpha, [N_sim, n_coef])+1)*xm

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

Emu = np.ones(n_coef)*mu
Emu = np.ones(n_coef)*xm*alpha/(alpha-1)

x, y = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
X = r_fun_np(x,y,Emu)*np.sin(x)*np.cos(y)
Y = r_fun_np(x,y,Emu)*np.sin(x)*np.sin(y)
EZ = r_fun_np(x,y,Emu)*np.cos(x)
ax.plot_surface(X, Y, EZ, color='red', alpha=0.8)

for i in range(np.min((N_sim,10))):
    X = r_fun_np(x,y,coef[i])*np.sin(x)*np.cos(y)
    Y = r_fun_np(x,y,coef[i])*np.sin(x)*np.sin(y)
    Z = r_fun_np(x,y,coef[i])*np.cos(x)
    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.1)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$a\left(x^{1}\right)^{2}+b\left(x^{2}\right)^{2}$')
ax.set_title('Realisations of Stochastic Riemannian Manifolds')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend1, legend2], ['Expected Manifold', 'Realisations'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Curvature

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for l in range(np.min((N_sim,10))):
    sec = jnp.zeros((len(x),len(x)))
    sec = vmap(lambda x1, x2: vmap(lambda y1, y2: dif_geo_jax.sectional_curvature(jnp.array([y1, y2]), f_fun=lambda x: f_fun(x,coef[l])))(x1,x2))(x[1:],y[1:])
    
    ax.plot_surface(x[1:], y[1:], sec, color='cyan', alpha=0.4)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of Stochastic Riemannian Manifolds')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend1, legend2], ['Curvature for Expected Metric', 'Realised Curvature'], numpoints = 1)


plt.tight_layout()
plt.show()