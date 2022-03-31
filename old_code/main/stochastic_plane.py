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

from sympy.plotting import plot_implicit

import dif_geo_sympy

import sympy as sym

#%% Hyper-parameter

np.random.seed(2712)

sigma = 1.0
mu = 0.0
n_points = 100
grid0 = -2
grid1 = 2
N_sim = 10

x = np.linspace(grid0,grid1,n_points)
y = np.linspace(grid0,grid1,n_points)
X, Y = np.meshgrid(x, y)


x1, x2 = sym.symbols('x1 x2')
v1, v2 = sym.symbols('v1 v2')
x = sym.Matrix([x1, x2])
a, b = sym.symbols('a b')

param_fun = sym.Matrix([x1, x2, a*x1+b*x2])

#%% Initial Computations

_ = dif_geo_sympy.compute_mmf(param_fun, x, print_latex = True)
_ = dif_geo_sympy.chris_symbols(x, param_fun=param_fun, print_latex=True)
_ = dif_geo_sympy.eq_geodesics(x, param_fun=param_fun, print_latex=True)
_ = dif_geo_sympy.sectional_curvature2d(x, param_fun=param_fun, print_latex=True)

#%% Plot realisations as well as expected parametrization (NOT EQUAL TO EXPECTED METRIC)

EZ = np.zeros((n_points, n_points))
coef = np.random.normal(mu, sigma, [N_sim, 2])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, EZ, color='red')

for i in range(np.min((N_sim,10))):
    Z = coef[i][1]*X+coef[i][1]*Y
    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.4)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$ax^{1}+bx^{2}$')
ax.set_title('Realisations of Stochastic Riemannian Manifolds')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend1, legend2], ['Expected Manifold', 'Realisations'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Expected Metric

EG = sym.Matrix([[2,0], [0,2]])
_ = dif_geo_sympy.chris_symbols(x, G=EG, print_latex=True)
_ = dif_geo_sympy.eq_geodesics(x, G=EG, print_latex=True)
_ = dif_geo_sympy.sectional_curvature2d(x, G=EG, print_latex=True)

#%% Stochastic Metric

SG, _ = dif_geo_sympy.compute_mmf(param_fun, x)

#%% Indicatrix

plt.rcParams['figure.figsize'] = 8,6
center = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

G = EG.subs(x1,center[0][0]).subs(x2,center[0][1])
ellipse_fun = lambda v1,v2: G[0,0]*(v1**2)+G[1,1]*(v2**2)+2*v1*v2*G[0,1]-1
plot1 = plot_implicit(ellipse_fun(v1,v2), (v1, -2, 2), (v2, -2, 2),
                      axis_center=(center[0][0], center[0][1]),
                      line_color='red', show=False)
for i in range(len(center)):
    G = EG.subs(x1,center[i][0]).subs(x2,center[i][1])
    ellipse_fun = lambda v1,v2: G[0,0]*(v1**2)+G[1,1]*(v2**2)+2*v1*v2*G[0,1]-1
    plot1 = plot_implicit(ellipse_fun(v1,v2), (v1, -2, 2), (v2, -2, 2),
                          axis_center=(center[i][0], center[i][1]),
                          line_color='red', show=False,
                          xlabel=r'$v^{1}$', ylabel=r'$v^{2}$')
    for j in range(N_sim):
        G = SG.subs(a,coef[j][0]).subs(b,coef[j][1])
        G = G.subs(x1,center[i][0]).subs(x2,center[i][1])
        ellipse_fun = lambda v1,v2: G[0,0]*(v1**2)+G[1,1]*(v2**2)+2*v1*v2*G[0,1]-1
        plot2 = plot_implicit(ellipse_fun(v1,v2), (v1, -2, 2), (v2, -2, 2),
                              axis_center=(center[i][0], center[i][1]),
                              line_color='cyan', alpha=0.2, show=False)
        plot1.append(plot2[0])
    
    plot1.show()
    
#%% Curvature

sec = dif_geo_sympy.sectional_curvature2d(x, G=EG)
if sec == 0.0:
    Esec = np.zeros((n_points, n_points))
else:
    Esec = sec(X,Y)

sec = dif_geo_sympy.sectional_curvature2d(x, G=SG)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Esec, color='red')

for i in range(np.min((N_sim,10))):
    sec_coef = sec.subs(a, coef[i][0]).subs(b, coef[i][1])
    sec_coef = sym.lambdify(x, sec_coef, modules='numpy')
    if sec == 0.0:
        sec_SG = np.zeros((n_points, n_points))
    else:
        sec_SG = sec_coef(X,Y)
    ax.plot_surface(X, Y, sec_SG, color='cyan', alpha=0.4)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of Stochastic Riemannian Manifolds')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend1, legend2], ['Curvature for Expected Metric', 'Realised Curvature'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Geodesics

n_grid = 100
y_init_grid = np.zeros((4, n_grid))
y0 = np.array([1.0,1.0])
yT = np.array([-1.0,1.0])
yEG, v = dif_geo_sympy.bvp_geodesic(y0, yT, n_grid, y_init_grid, x, G = EG)

fig, ax = plt.subplots(1,2,figsize=(8,6))
ax[0].plot(np.linspace(0,1,yEG.shape[-1]), yEG[0], color='red')
ax[1].plot(np.linspace(0,1,yEG.shape[-1]), yEG[1], color='red')
for i in range(N_sim):
    G = SG.subs(a, coef[i][0]).subs(b, coef[i][1])
    ySG, v = dif_geo_sympy.bvp_geodesic(y0, yT, n_grid, y_init_grid, x, G = G)
    ax[0].plot(np.linspace(0,1,ySG.shape[-1]), ySG[0], color='cyan', alpha=0.4)
    ax[1].plot(np.linspace(0,1,ySG.shape[-1]), ySG[1], color='cyan', alpha=0.4)

ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()



