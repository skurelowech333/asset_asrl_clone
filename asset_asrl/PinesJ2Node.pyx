# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:51:31 2026

@author: Sarah
"""

# =========================================================
# FILE: PinesJ2Node.pyx
# =========================================================


import asset_asrl as ast

from pines_kernel import accel_j2


vf = ast.VectorFunctions



cdef class PinesJ2Node:

    cdef:

        object x
        object y
        object z

        double mu
        double Re
        double J2



    def __cinit__(
        self,
        object x,
        object y,
        object z,
        double mu,
        double Re,
        double J2):


        self.x=x
        self.y=y
        self.z=z

        self.mu=mu
        self.Re=Re
        self.J2=J2



    def eval(self):

        return accel_j2(
            float(self.x),
            float(self.y),
            float(self.z),
            self.mu,
            self.Re,
            self.J2
        )