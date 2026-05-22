# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:30:39 2026

@author: Sarah
"""

import numpy as np
import asset_asrl as ast
vf = ast.VectorFunctions
oc = ast.OptimalControl

from asset_asrl.Forces.GravityRHS import GravityRHS


# =========================================================
# WRAP C KERNEL INTO ASSET VECTORFUNCTION
# =========================================================
class GravityCythonVF:

    def __init__(self, mu):

        self.kernel = GravityRHS(mu)

        # dummy VF placeholder (NOT symbolic physics)
        X = vf.Arguments(6)

        self._vf = vf.external(self._eval, 6, 6)

    # -----------------------------------------------------
    # CALLED BY ASSET DURING INTEGRATION
    # -----------------------------------------------------
    def _eval(self, x):

        out = np.zeros(6)

        self.kernel.eval(
            x[0], x[1], x[2],
            x[3], x[4], x[5],
            out
        )

        return out

    # -----------------------------------------------------
    # RETURN VF HANDLE
    # -----------------------------------------------------
    def vf(self):
        return self._vf