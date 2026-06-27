# -*- coding: utf-8 -*-
"""
build.py

Rebuild all Cython extensions in-place for local development

@author: Sarah
"""

import sys
import subprocess
import platform
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def print_header(title):

    print("\n===================================")
    print(title)
    print("===================================\n")

def main():

    print_header("CYTHON BUILD STARTING")


    print("Python:")
    print(sys.executable)

    print("\nVersion:")
    print(sys.version)

    print("\nPlatform:")
    print(platform.platform())



    cmd = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace",
        "--force"
    ]


    print("\nRunning:")
    print(" ".join(cmd))
    print()



    try:

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True
        )


        if result.stdout:

            print("\n--- BUILD OUTPUT ---")
            print(result.stdout)



        if result.stderr:

            print("\n--- BUILD ERRORS ---")
            print(result.stderr)



        #show_extensions()



        if result.returncode == 0:


            print_header(
                "BUILD SUCCESS"
            )


            pyds = list(ROOT.rglob("*.pyd"))


            if not pyds:

                print(
                    "WARNING:"
                    "\nBuild returned success but no .pyd files found."
                    "\nCheck setup.py extension list."
                )


            return 0




        print_header(
            "BUILD FAILED"
        )


        print(
            f"Exit code: {result.returncode}"
        )


        err = result.stderr.lower()

        if (
            "no module named cython" in err
            or
            "modulenotfounderror" in err
        ):

            print(
                "\nERROR:"
                "\nCython package missing."
                "\nRun:"
                "\n  pip install cython"
            )



        elif "cl.exe" in err:

            print(
                "\nERROR:"
                "\nMSVC compiler missing."
            )



        elif (
            "expected an increase" in err
            or
            "indentation" in err
        ):

            print(
                "\nERROR:"
                "\nCython indentation error."
                "\nCheck whitespace around def/cdef blocks."
            )



        elif "cannot open include file" in err:

            print(
                "\nERROR:"
                "\nMissing C/C++ include file."
            )



        elif "python.h" in err:

            print(
                "\nERROR:"
                "\nPython development headers missing."
            )



        elif "undeclared name" in err:

            print(
                "\nERROR:"
                "\nCython symbol/import problem."
                "\nCheck imports and function visibility."
            )



        else:

            print(
                "\nERROR:"
                "\nUnknown compiler failure."
                "\nSee output above."
            )


        return result.returncode


    except Exception:

        print_header(
            "BUILD SCRIPT CRASHED"
        )

        traceback.print_exc()

        return 1

if __name__ == "__main__":

    sys.exit(main())