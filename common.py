# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:50:06 2026

@author: Sarah
"""

""" 
Colors and logging functions
"""
import unittest

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def log(msg, color=C.BLUE):
    print(color + msg + C.RESET)


def ok(msg):
    print(C.GREEN + msg + C.RESET)


def warn(msg):
    print(C.YELLOW + msg + C.RESET)


def fail(msg):
    print(C.RED + msg + C.RESET)

""" 
Constants
"""
MU_EARTH = 398600.435507
R_EARTH = 6378.137

Cd = 2.2
A_over_m = 0.01
rho0 = 1.225

"""
Summary runner class
"""
class SummaryRunner(unittest.TextTestRunner):

    def run(self, test, name):
        result = super().run(test)

        total = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = total - failures - errors

        print("\n" + "=" * 60)
        print(f"{name} KERNEL TEST SUMMARY")
        print("=" * 60)

        print(f"Total tests : {total}")
        print(f"Passed      : {passed}")
        print(f"Failures    : {failures}")
        print(f"Errors      : {errors}")

        if failures == 0 and errors == 0:
            print(C.GREEN + f"\nALL {name} TESTS PASSED" + C.RESET)
        else:
            print(C.RED + f"\nSOME {name} TESTS FAILED" + C.RESET)

        print("=" * 60)

        return result