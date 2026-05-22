# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:42:48 2026

@author: Sarah
"""

import asset_asrl as ast

vf = ast.VectorFunctions
Args = vf.Arguments


class GravitySystem:

    def __init__(self, kernels):
        self.kernels = kernels
        self._ode = self._build()

    def _build(self):

        X = Args(6)
        v = X.tail3()

        acc = None

        for k in self.kernels:

            if acc is None:
                acc = k.vf()
            else:
                acc = acc + k.vf()

        return vf.stack(v, acc)

    def ode(self):
        return self._ode

    def __call__(self):
        return self._ode