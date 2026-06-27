import asset_asrl as ast

vf = ast.VectorFunctions


def Gradient3(
    scalar,
    x
):

    return vf.stack(
        [
            vf.diff(scalar,x[0]),
            vf.diff(scalar,x[1]),
            vf.diff(scalar,x[2])
        ]
    )