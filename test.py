# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:39:21 2026

@author: Sarah
"""

from PinesVW import BuildVW

V,W = BuildVW(
    7000.0,
    0.0,
    0.0,
    6378.137,
    2
)

print(sorted(V.keys()))