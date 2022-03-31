#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:01:52 2022

@author: root
"""

#%% Modules

#Adds the source library to Python Path
import sys
sys.path.insert(1, '../src/')

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.animation as animation

from dif_geo_sympy import rm_geometry

import sympy as sym

#%% Functions

def f_poly_1d(x,coef):
    
    n_poly = len(coef)
    z = 0.0
    for i in range(n_poly):
        z += coef[i]*(x**i)
        
    return z

def f_poly_2d(x,y,coef):
    
    n_poly = len(coef)
    z = 0.0
    for i in range(n_poly):
        for j in range(n_poly):
            z += coef[i,j]*(x**i)*(y**j)
            
    return z
    

#%% Hyper-parameters

np.random.seed(2712)

rm = rm_geometry()
n_poly = 2 #If 4, then y=a*x**3+b*x**2+c*x**1+d*x**0
sigma = 1.0
mu = 0.0
n_points = 100
grid0 = -1
grid1 = 1
N_sim = 5

x = np.linspace(grid0,grid1,n_points)
y = np.linspace(grid0,grid1,n_points)

coef = np.random.normal(mu, sigma, [N_sim, n_poly, n_poly])

z = f_poly_2d(X, Y, coef[0])

#%% Computing geodesics for samples of metrics

x1, x2 = sym.symbols('x1 x2')
coef = np.random.normal(mu, sigma, [N_sim, n_poly, n_poly])
x = sym.Matrix([x1, x2])
param_fun = sym.Matrix([x1, x2, f_poly_2d(x1, x2, coef[0])])
        
G = sym.simplify(rm.compute_mmf(param_fun, x))
G_inv = sym.simplify(rm.get_immf())
chris = sym.simplify(rm.get_christoffel_symbols())
geodesic_equation = rm.get_geodesic_equation_2d()
parallel_equation = rm.get_parallel_transport_equation_2d()


G_inv = np.array(test.get_immf())
G = np.array(test.get_mmf())

#3,3,0
#-3,3,0
y_init = np.zeros((4, 100))

y = rm.bvp_geodesic(np.array([3,3]), np.array([-3,3]), 100, y_init)
y = test.ivp_geodesic(10, [3,3,-7.49514614,   4.50945609])
v = test.parallel_transport_along_geodesic(np.array([3,3]), np.array([-3,3]), np.array([0,1]), 100)

x1 = np.linspace(-5,5,100)
x2 = np.linspace(-5,5,100)
x3 = x1**2-x2**2
X = np.vstack((x1,x2))

val = test.karcher_mean_algo(X)

#%% Simulating metric

coef = np.random.normal(mu, sigma, [N_sim, n_poly, n_poly])
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
z = f_poly_2d(X, Y, coef[0])

Z = z.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

