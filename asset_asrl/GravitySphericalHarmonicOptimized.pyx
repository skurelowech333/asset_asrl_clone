# ============================================================
# FILE: GravitySphericalHarmonic.pyx
#
# Classical spherical harmonic gravity kernel
#
# Fair comparison with Pines:
#   - Same Legendre
#   - Same coefficients
#   - Same normalization
#   - Same trig recurrence
#   - Same spherical conversion
#
# ============================================================


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


import asset_asrl as ast

cimport cython


from Legendre import AssociatedLegendreGradient
from pines_math import (
    HarmonicTrig,
    SphericalToCartesian
)



vf = ast.VectorFunctions
Args = vf.Arguments



# ============================================================
# CLASSICAL SPHERICAL HARMONIC
# ============================================================


cdef class SphericalHarmonicGravityVF:


    cdef:

        double mu
        double radius

        int degree
        int N1

        #
        # Gravity parameters
        #
        object Cnm
        object Snm


        #
        # Flattened coefficient storage
        #
        object Cflat
        object Sflat


        #
        # Degree multiplier (n+1)
        #
        object Ncoef


        #
        # Non-zero harmonic indices
        #
        object harmonic_indices


        #
        # Cached ASSET graphs
        #
        object _acc_vf
        object _ode_vf



    # ========================================================
    # Constructor
    # ========================================================

    def __cinit__(
        self,
        double mu,
        double radius,
        object Cnm,
        object Snm,
        int degree
    ):


        cdef:

            int n
            int m
            int idx

            int count = 0



        print("Initializing spherical harmonic gravity")


        self.mu = mu
        self.radius = radius
        self.degree = degree

        self.N1 = degree + 1


        self.Cnm = Cnm
        self.Snm = Snm



        print("Allocating coefficient storage")


        #
        # Flat arrays
        #
        self.Cflat = {}

        self.Sflat = {}


        #
        # Degree constants
        #
        self.Ncoef = {}



        #
        # Store only non-zero harmonics
        #
        self.harmonic_indices = []



        print("Processing coefficients")


        for n in range(2, degree+1):


            self.Ncoef[n] = float(n+1)


            for m in range(n+1):


                idx = (
                    n*self.N1
                    +
                    m
                )


                C = float(
                    Cnm[n,m]
                )


                S = float(
                    Snm[n,m]
                )



                #
                # Skip empty terms
                #
                if (
                    C == 0.0
                    and
                    S == 0.0
                ):
                    continue



                self.Cflat[idx] = C

                self.Sflat[idx] = S



                self.harmonic_indices.append(
                    (
                        n,
                        m,
                        idx
                    )
                )


                count += 1



        print(
            f"Stored {count} harmonic terms"
        )



        print(
            "Building acceleration graph"
        )


        self._acc_vf = (
            self._build_acceleration()
        )


        print(
            "Acceleration graph complete"
        )



        self._ode_vf = (
            self._build_ode()
        )


        print(
            "ODE graph complete"
        )



    # ========================================================
    # Acceleration
    # ========================================================


    cdef object _build_acceleration(self):


        cdef int n
        cdef int m
        cdef int idx
    
    
        print("Creating ASSET arguments")
    
        X = Args(6)
    
    
        rvec = X.head3()
    
    
        x = rvec[0]
        y = rvec[1]
        z = rvec[2]
    
    
    
        # ========================================================
        # Geometry
        # ========================================================
    
        print("Building geometry")
    
    
        r2 = (
            x*x
            +
            y*y
            +
            z*z
        )
    
    
        r = r2.sqrt()
    
    
        inv_r = (
            X.Constant(1.0)
            /
            r
        )
    
    
        xy2 = (
            x*x
            +
            y*y
        )
    
    
        xy = xy2.sqrt()
    
    
        inv_xy = (
            X.Constant(1.0)
            /
            xy
        )
    
    
    
        sinphi = (
            z
            *
            inv_r
        )
    
    
        cosphi = (
            xy
            *
            inv_r
        )
    
    
        coslam = (
            x
            *
            inv_xy
        )
    
    
        sinlam = (
            y
            *
            inv_xy
        )
    
    
    
        rho = (
            X.Constant(self.radius)
            *
            inv_r
        )
    
    
    
        # ========================================================
        # Harmonic helpers
        # ========================================================
    
        print("Building trig recurrence")
    
    
        Ctrig, Strig = HarmonicTrig(
            coslam,
            sinlam,
            self.degree
        )
    
    
        print("Building Legendre")
    
    
        P, dP = AssociatedLegendreGradient(
            sinphi,
            self.degree
        )
    
    
    
        # ========================================================
        # rho powers
        # ========================================================
    
        print("Building rho cache")
    
    
        rho_cache = {}
    
    
        rho_n = X.Constant(1.0)


        for n in range(1,self.degree+1):
        
            rho_n = rho_n*rho
        
            rho_cache[n]=rho_n
            
    
    
        # ========================================================
        # Harmonic sums
        # ========================================================
    
        print(
            "Building harmonic terms:",
            len(self.harmonic_indices)
        )
    
    
        radial_terms = []
    
        latitude_terms = []
    
        longitude_terms = []
    
    
    
        for n,m,idx in self.harmonic_indices:
    
    
            Pnm = P[idx]
    
            dPnm = dP[idx]
    
    
    
            #
            # Cached coefficients
            #
    
            C = X.Constant(
                self.Cflat[idx]
            )
    
    
            S = X.Constant(
                self.Sflat[idx]
            )
    
    
    
            #
            # Trig combinations
            #
    
            Ct = Ctrig[m]
    
            St = Strig[m]
    
    
            Cct = (
                C
                *
                Ct
            )
    
    
            Sst = (
                S
                *
                St
            )
    
    
            trig = (
                Cct
                +
                Sst
            )
    
    
            dtrig = (
                m
                *
                (
                    S*Ct
                    -
                    C*St
                )
            )
    
    
    
            rn = rho_cache[n]
    
    
    
            #
            # radial component
            #
    
            radial_terms.append(
                X.Constant(
                    self.Ncoef[n]
                )
                *
                rn
                *
                Pnm
                *
                trig
            )
    
    
    
            #
            # latitude component
            #
    
            latitude_terms.append(
                rn
                *
                dPnm
                *
                trig
            )
    
    
    
            #
            # longitude component
            #
    
            longitude_terms.append(
                rn
                *
                Pnm
                *
                dtrig
            )
    
    
    
        print("Combining sums")
    
    
        radial_sum = (
            X.Constant(1.0)
            +
            balanced_sum(radial_terms)
        )
    
    
        latitude_sum = balanced_sum(
            latitude_terms
        )
    
    
        longitude_sum = balanced_sum(
            longitude_terms
        )
    
    
    
        # ========================================================
        # Spherical acceleration
        # ========================================================
    
    
        print("Building acceleration conversion")
    
    
        mu_r2 = (
            X.Constant(self.mu)
            /
            r2
        )
    
    
    
        ar = (
            -mu_r2
            *
            radial_sum
        )
    
    
        aph = (
            mu_r2
            *
            latitude_sum
        )
    
    
        alam = (
            mu_r2
            *
            longitude_sum
            /
            cosphi
        )
    
    
    
        ax,ay,az = SphericalToCartesian(
            ar,
            aph,
            alam,
            sinphi,
            cosphi,
            sinlam,
            coslam
        )
    
    
    
        print("Stacking acceleration")
    
    
        return vf.stack(
            [
                ax,
                ay,
                az
            ]
        )



    # ========================================================
    # ODE
    # ========================================================


    cdef object _build_ode(self):


        X = Args(6)

        v = X.tail3()

        a = self._acc_vf


        return vf.stack(
            [
                v[0],
                v[1],
                v[2],

                a[0],
                a[1],
                a[2]
            ]
        )



    # ========================================================
    # API
    # ========================================================


    cpdef object vf(self):

        return self._acc_vf


    cpdef object build_accel(self):

        return self._acc_vf


    cpdef object ode(self):

        return self._ode_vf


    def __call__(self):

        return self._ode_vf



    def __repr__(self):

        return (
            f"SphericalHarmonicGravityVF("
            f"degree={self.degree})"
        )
    
def balanced_sum(items):

    if len(items) == 0:
        return None

    while len(items) > 1:

        reduced = []

        for i in range(0,len(items),2):

            if i+1 < len(items):

                reduced.append(
                    items[i] + items[i+1]
                )

            else:

                reduced.append(
                    items[i]
                )

        items = reduced

    return items[0]