#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:26:13 2019

@author: cyq
"""
from sympy import S, log
from sympy.calculus import finite_diff_weights
import numpy as np
from sympy import Function, hessian, pprint
from sympy.abc import x, y,Y
from sympy import symbols, sin, cos, pi
from sympy.diffgeom import Manifold, Patch, CoordSystem
from sympy.simplify import simplify
from sympy.physics.vector import gradient
from sympy.physics.vector import ReferenceFrame
from sympy import *
from sympy.tensor import IndexedBase, Idx
from sympy import symbols
from sympy import Matrix, I
from sympy import symbols, sin, cos, pi
from sympy.diffgeom import Manifold, Patch, CoordSystem
if __name__ == '__main__':
    
    #f = Function('f')(x, y)
    #g1 = Function('g')(x, y)
    a1,a2,a3,ao1,ao2,theta1,theta2,theta3,theta4 = symbols('a1,a2,a3,ao1,ao2,theta1,theta2,theta3,theta4')
    m = Manifold('M', 4)
    patch = Patch('P', m)
    rect = CoordSystem('rect', patch)
    polar = CoordSystem('polar', patch)
    rect in patch.coord_systems
    polar.connect_to(rect, [theta1,theta2,theta3,theta4], [a1*theta1+a2*theta2+a3*theta3+(theta1*ao1+theta2*ao2)*theta4])
    print(polar.jacobian(rect, [theta1,theta2,theta3,theta4]))
    g=polar.jacobian(rect, [theta1,theta2,theta3,theta4])
    
    patch2 = Patch('P', m)
    rect2 = CoordSystem('rect2', patch2)
    polar2 = CoordSystem('polar2', patch2)
    rect2 in patch2.coord_systems
    polar2.connect_to(rect2, [theta1,theta2,theta3,theta4], g)
    print(polar2.jacobian(rect2, [theta1,theta2,theta3,theta4]))
    H=polar2.jacobian(rect2, [theta1,theta2,theta3,theta4])
'''
    n = Manifold('M', 1)
    patch3 = Patch('P', n)
    rect3 = CoordSystem('rect3', patch3)
    polar3 = CoordSystem('polar3', patch3)
    rect3 in patch3.coord_systems
    polar3.connect_to(rect3, [theta1], g)
    print(polar3.jacobian(rect3, [theta1]))
    gd=polar3.jacobian(rect3, [theta1])
    
    patch4 = Patch('P', n)
    rect4 = CoordSystem('rect4', patch4)
    polar4 = CoordSystem('polar4', patch4)
    rect4 in patch4.coord_systems
    polar4.connect_to(rect4, [theta1], H)
    print(polar4.jacobian(rect4, [theta1]))
    Hdd=polar4.jacobian(rect4, [theta1])
    Hd=Hdd.reshape(2,2)
    #i=H[0].shape()
    I=eye(2)
    #I=eye(H.shape()[0])
    #print(I.shape(I)[0])
    M=H**-1*Hd
    #print("特征值与特征向量: ", M.eigenvals())
    c=np.trace(M)
    print(c)
    reg=[1e-1]
    H = H-reg[0]*I
    h=H**-1*g.T
    grad=h.T*gd-0.5*h.T*Hd*h+0.5*Matrix([[c]])
    
    print (grad)
    
    #L = g*H**-1*g.T+log(abs(-H))
    #pprint(L)
    
    H=hessian(g, ())
    car = x**2 + 3*y 
    g=car.diff(x)
    H=hessian(g, (x, y))
    I=eye(H.shape[0])
    
    reg=[1e-1]
    H = H-reg[0]*I
    L = g.T*H**-1*g+log(-H)
    pprint(hessian(car, (x, y)))
    pprint(gradient(car, x))
    pprint(L)
    
    r = car.traj.reward(reward)#car means 
    g = utils.grad(r, car.traj.u)
    H = utils.hessian(r, car.traj.u)
    I = tt.eye(utils.shape(H)[0])
    reg = utils.vector(1)
    reg.set_value([1e-1])
    H = H-reg[0]*I
    L = tt.dot(g, tt.dot(tn.MatrixInverse()(H), g))+tt.log(tn.Det()(-H))
    

from sympy import symbols, sin, cos
from sympy.diffgeom import Manifold, Patch, CoordSystem
#from sympy.simplify import simplify
from sympy import Function, hessian, pprint
if __name__ == '__main__':
    r, theta = symbols('r, theta')
    m = Manifold('M', 2)
    patch = Patch('P', m)
    rect = CoordSystem('rect', patch)
    polar = CoordSystem('polar', patch)
    rect in patch.coord_systems
    f = Function('f')(r, theta)
    polar.connect_to(rect, [r, theta], [r*cos(theta), r*sin(theta)])
    pprint(polar.jacobian(rect, [r, theta]))
    pprint(hessian(f, (r, theta), polar))
'''