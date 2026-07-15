cdef class GravityCoordinates:

    cdef:
        object r
        object rho
        object u
        object v
        object w

    def __init__(self, object X, double Re):

        pos = X.head3()

        self.r = pos.norm()

        self.rho = X.Constant(Re) / self.r

        self.u = pos[0] / self.r
        self.v = pos[1] / self.r
        self.w = pos[2] / self.r