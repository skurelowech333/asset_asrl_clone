# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:50:18 2026

@author: Sarah
"""

import numpy as np


class GravityCoefficients:

    def __init__(self, degree):

        self.degree = degree

        self.C = np.zeros(
            (degree+1, degree+1),
            dtype=float
        )

        self.S = np.zeros(
            (degree+1, degree+1),
            dtype=float
        )


    def set(self,n,m,Cnm,Snm=0.0):

        self.C[n,m] = Cnm
        self.S[n,m] = Snm