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

import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy 

from sympy.plotting import plot_implicit

import dif_geo_sympy

import sympy as sym

#%% Hyper-parameters

mu = 1.0
sigma = 0.1
N_sim = 1
m = 1
l = 1

#%% Initial Computations

#_ = dif_geo_sympy.compute_mmf(param_fun, x, print_latex = True)
#_ = dif_geo_sympy.chris_symbols(x, param_fun=param_fun, print_latex=True)
#_ = dif_geo_sympy.eq_geodesics(x, param_fun=param_fun, print_latex=True)
#_ = dif_geo_sympy.sectional_curvature2d(x, param_fun=param_fun, print_latex=True)

#%% Plot realisations as well as expected parametrization (NOT EQUAL TO EXPECTED METRIC)

coef = np.random.normal(mu, sigma, [N_sim,3])
alpha = 5.0
xm = 1.0
coef = (np.random.pareto(alpha, [N_sim,3])+1)*xm

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

Emu = mu
Emu = xm*alpha/(alpha-1)

x, y = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
X = np.real(scipy.special.sph_harm(m,l,x,y))*np.sin(x)*np.cos(y)
Y = np.real(scipy.special.sph_harm(m,l,x,y))*np.sin(x)*np.sin(y)
EZ = np.real(scipy.special.sph_harm(m,l,x,y))*np.cos(x)
ax.plot_surface(X, Y, EZ, color='red', alpha=0.8)

for i in range(np.min((N_sim,10))):
    X = coef[i,0]*np.real(scipy.special.sph_harm(m,l,x,y))*np.sin(x)*np.cos(y)
    Y = coef[i,1]*np.real(scipy.special.sph_harm(m,l,x,y))*np.sin(x)*np.sin(y)
    Z = coef[i,2]*np.real(scipy.special.sph_harm(m,l,x,y))*np.cos(x)
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
