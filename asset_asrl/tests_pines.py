# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:51:55 2026

@author: Sarah
"""

import numpy as np

from GravityPines import PinesGravityVF


MU = 398600.4418
RE = 6378.1363

J2 = 1.08262668e-3



g = PinesGravityVF(
    MU,
    RE,
    J2
)


r = np.array(
    [
        7000,
        0,
        0
    ]
)


print(g)

print(
    g.vf()
)