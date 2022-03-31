# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 11:53:35 2022

@author: Frederik
"""

#%% Modules used

#Numerical python
import numpy as np

#scipy
from scipy.integrate import solve_bvp, solve_ivp

#Fpr symbolic computations
import sympy as sym

#%% Functions

def compute_mmf(param_fun, x_sym, print_latex = False):
    
    def print_fun_latex(M, M_string):
        
        print("\n")
        print(r"\begin{equation}")
        print("\t", M_string, " = ")
        print("\t",r"\begin{pmatrix}")
        for i in range(dim):
            if i>0:
                print(r' \\', '\n')
            for j in range(dim):
                print("\t\t", sym.latex(M[i,j]), end='')
                if j < dim-1:
                    print("\t &", end='')
                  
        print("\n\t",r"\end{pmatrix}")
        print(r"\end{equation}")
                
    
    jacobian = param_fun.jacobian(x_sym)
    G = (jacobian.T)*jacobian
    G_inv = G.inv()
    
    if print_latex:
        G = sym.simplify(G)
        G_inv = sym.simplify(G_inv)
        dim = len(x_sym)
        print_fun_latex(G, 'G')
        print_fun_latex(G_inv, 'G^{-1}')
        
    
    return G, G_inv

def chris_symbols(x_sym, param_fun = None, G = None, print_latex=False):
    
    def print_fun_latex():
        
        print("\n")
        print(r"\begin{equation}")
        print("\t",r"\begin{split}")
        for i in range(dim):
            for j in range(dim):
                for m in range(dim):
                    print("\t\t\Gamma_{", str(i+1), ",", str(j+1), "}^{", str(m+1), "} &= ", 
                          sym.latex(christoffel[i][j][m]), r'\\')
        print("\t",r"\end{split}")
        print(r"\end{equation}")
    
    
    if G is None:
        G, G_inv = compute_mmf(param_fun, x_sym)
    else:
        G_inv = G.inv()
    
    dim = len(x_sym)
    
    christoffel = np.zeros((dim, dim, dim), dtype=object)
    
    for i in range(dim):
        for j in range(dim):
            for m in range(dim):
                for l in range(dim):
                    christoffel[i][j][m] += (sym.diff(G[j,l],x_sym[i])+
                                             sym.diff(G[l,i], x_sym[j])-
                                             sym.diff(G[i,j], x_sym[l]))*G_inv[l,m]
                christoffel[i][j][m] /= 2
    
    if print_latex:
        christoffel = sym.simplify(christoffel)
        print_fun_latex()
    
    return christoffel

def eq_geodesics(x_sym, param_fun = None, G = None, print_latex=False):
    
    def print_fun_latex():
        
        print("\n")
        print(r"\begin{equation}")
        print("\t",r"\begin{split}")
        for i in range(dim):
            print("\t\t",  r'\ddot{\gamma}', '^{', str(i+1), '}+', sym.latex(eq[0]), ' &= 0', r'\\')
        print("\t",r"\end{split}")
        print(r"\end{equation}")
    
    dim = len(x_sym)
    if G is None:
        G, _ = compute_mmf(param_fun, x_sym)
        
    christoffel = chris_symbols(x_sym, G = G)
    gamma = sym.symbols('\gamma_1:{}'.format(dim+1))
    gamma_diff = sym.symbols('d\gamma_1:{}'.format(dim+1))
    christoffel = sym.lambdify(x_sym, christoffel, modules='numpy')
    chris = christoffel(*gamma)
    eq = []
    
    for k in range(dim):
        eq1 = 0.0
        for i in range(dim):
            for j in range(dim):
                eq1 += gamma_diff[i]*gamma_diff[j]*chris[i][j][k]
            
        eq.append(sym.simplify(eq1))
        
    if print_latex:
        print_fun_latex()

    return eq

def eq_pt(x_sym, param_fun = None, G = None, print_latex=False):
        
    ### PARALLEL TRANSPORT
    
    def print_fun_latex():
        
        print("\n")
        print(r"\begin{equation}")
        print("\t",r"\begin{split}")
        for i in range(dim):
            print("\t\t", sym.latex(eq[0]), ' &= 0', r'\\')
        print("\t",r"\end{split}")
        print(r"\end{equation}")
    
    dim = len(x_sym)
    if G is None:
        G, _ = compute_mmf(param_fun, x_sym)
    
    v = sym.symbols('v_1:{}'.format(dim+1))
    gamma = sym.symbols('\gamma_1:{}'.format(dim+1))
    gamma_diff = sym.symbols('d\gamma_1:{}'.format(dim+1))
    
    christoffel = chris_symbols(x_sym, G = G)
    christoffel = sym.lambdify(x_sym, christoffel, modules='numpy')
    chris = christoffel(*gamma)
    
    eq = []
    for k in range(dim):
        eq1 = 0.0
        for i in range(dim):
            for j in range(dim):
                eq1 += v[j]*gamma_diff[i]*chris[i][j][k]
        eq.append(sym.simplify(eq1))
    
    if print_latex:
        print_fun_latex()
    
    return eq

def sectional_curvature2d(x_sym, param_fun = None, G = None, print_latex=False):
    
    def print_fun_latex():
        
        print("\n")
        print(r"\begin{equation}")
        print("\t", 'K = ', sym.latex(eq))
        print(r"\end{equation}")
    
    dim = len(x_sym)
    if G is None:
        G, _ = compute_mmf(param_fun, x_sym)
    
    chris = chris_symbols(x_sym, G = G)
    
    eq = 0.0
    for s in range(dim):
        eq1 = 0.0
        for p in range(dim):
            eq1 += chris[1,1,p]*chris[0,p,s]-chris[0,1,p]*chris[1,p,s]
        eq += (sym.diff(chris[1,1,s], x_sym[0])-sym.diff(chris[0,1,s], x_sym[1])+eq1)*G[s,0]
    
    if print_latex:
        eq = sym.simplify(eq/G.det())
        print_fun_latex()
    
    return eq

def bvp_geodesic(y0, yT, n_grid, y_init_grid, x_sym, param_fun = None, G = None):
    
    def geodesic_equation_bc(ya, yb):
        
        bc = []
        
        for i in range(dim):
            bc.append(ya[i]-y0[i])
            bc.append(yb[i]-yT[i])
            
        return bc   
    
    def geodesic_equation_fun(t, y):
        
        gamma = np.array(y[0:dim])
        gamma_diff = np.array(y[dim:])
        
        chris = np.array(christoffel(*gamma), dtype=object)
            
        dgamma = gamma_diff
        dgamma_diff = np.zeros(gamma_diff.shape)
        
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    dgamma_diff[k] = dgamma_diff[k]+gamma_diff[i]*gamma_diff[j]*chris[i][j][k]
            dgamma_diff[k] = -dgamma_diff[k]
    
        return np.concatenate((dgamma, dgamma_diff))
    
    dim = len(x_sym)
    if G is None:
        G, _ = compute_mmf(param_fun, x_sym)
    
    chris_sym = chris_symbols(x_sym, G = G)
    christoffel = sym.lambdify(x_sym, chris_sym, modules='numpy')
        
    x_mesh = np.linspace(0,1, n_grid)
    
    sol = solve_bvp(geodesic_equation_fun, 
                    geodesic_equation_bc, 
                    x_mesh, y_init_grid)
    
    y = sol.y[0:dim]
    v = sol.y[dim:]
    
    return y, v




