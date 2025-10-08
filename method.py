from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
from mmapy import mmasub, subsolv


# THIS FILE FOR RAMP AND BESO
# element stiffness matrix
def lk():
    E = 1
    nu = 0.3
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    KE = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
    return KE


# Optimality criterion
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
    # reshape to perform vector operations
    xnew = np.zeros(nelx * nely)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew[:] = np.maximum(
            0.0,
            np.maximum(
                x - move,
                np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))),
            ),
        )
        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)


def main(
    nelx, nely, volfrac, penal, rmin, ft, xsolv, method="SIMP", ramp_q=0.5, b_erase=0.02
):
    """
    Main driver with three method options:
      method = 'SIMP'  : original SIMP (default)
      method = 'RAMP'  : RAMP interpolation
      method = 'BESO'  : simple BESO / ESO-style evolutionary method (binary)
    ramp_q: RAMP parameter q (typical 0.5 -- try 0.2..2.0)
    b_erase: BESO evolution rate (fraction of elements to add/remove each iter)
    """
    print("Topology optimization - method:", method)
    print("ndes: " + str(nelx) + " x " + str(nely))
    print(
        "volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal)
    )
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    print("Optimizer: " + ["OC method", "MMA"][xsolv])

    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0

    # dofs:
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Allocate design variables (as array), initialize and allocate sens.
    n = nely * nelx

    # initial designs:
    if method.upper() == "BESO":
        # start from full material for BESO (typical ESO/BESO approach)
        x = np.ones(n, dtype=float)
    else:
        # SIMP / RAMP use continuous initial design at volfrac
        x = volfrac * np.ones(n, dtype=float)

    xPhys = x.copy()
    dc = np.zeros((nely, nelx), dtype=float).flatten()  # flattened later if needed

    # Initialize OC
    if xsolv == 0:
        xold1 = x.copy()
        g = 0  # must be initialized to use the Nguyen/Paulino OC approach
    # Initialize MMA
    elif xsolv == 1:
        m = 1
        xmin = np.zeros((n, 1))
        xmax = np.ones((n, 1))
        xval = x[:, None]
        xold1 = xval.copy()
        xold2 = xval.copy()
        low = np.ones((n, 1))
        upp = np.ones((n, 1))
        a0 = 1.0
        a = np.zeros((m, 1))
        c = 10000 * np.ones((m, 1))
        d = np.zeros((m, 1))
        move = 0.2

    # FE: Build the index vectors for the coo matrix format.
    KE = lk()
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [
                    2 * n1 + 2,
                    2 * n1 + 3,
                    2 * n2 + 2,
                    2 * n2 + 3,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n1,
                    2 * n1 + 1,
                ]
            )

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # BC's and support
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    fixed = np.union1d(
        dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1])
    )
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load
    f[1, 0] = -1

    # Set loop counter and gradient vectors
    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    # For BESO control
    if method.upper() == "BESO":
        # target number of solid elements
        n_target = int(np.round(volfrac * n))
        # evolution rate -> absolute number change per iteration
        n_erase_per_iter = max(1, int(np.round(b_erase * n)))

    # Main loop
    while (change > 0.001) and (loop < 2000):
        loop = loop + 1

        # -------------------------
        #  FE: build stiffness matrix
        # -------------------------
        if method.upper() == "SIMP":
            E_elems = Emin + xPhys**penal * (Emax - Emin)
        elif method.upper() == "RAMP":
            q = ramp_q
            denom = 1.0 + q * (1.0 - xPhys)
            E_elems = Emin + (xPhys / denom) * (Emax - Emin)
        elif method.upper() == "BESO":
            # binary material model: xPhys is 0/1 (but keep Emin for stability)
            E_elems = Emin + xPhys * (Emax - Emin)
        else:
            raise ValueError("Unknown method: " + str(method))

        # Build global K (same style as original)
        sK = (KE.flatten()[:, None] * (E_elems)).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])

        # -------------------------
        # Objective and element sensitivities
        # -------------------------
        ce[:] = (
            np.dot(u[edofMat].reshape(nelx * nely, 8), KE)
            * u[edofMat].reshape(nelx * nely, 8)
        ).sum(1)
        obj = (E_elems * ce).sum()

        # Sensitivity (derivative dC/dx = - dE/dx * ce)
        if method.upper() == "SIMP":
            dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        elif method.upper() == "RAMP":
            q = ramp_q
            dE_dx = (Emax - Emin) * (1.0 + q) / (1.0 + q * (1.0 - xPhys)) ** 2
            dc[:] = -dE_dx * ce
        elif method.upper() == "BESO":
            # For BESO we'll use ce ranking; keep dc for filter compatibility
            dc[:] = -(Emax - Emin) * ce

        dv[:] = np.ones(nely * nelx)

        # -------------------------
        # Sensitivity filtering (keep original filtering behaviour)
        # -------------------------
        if ft == 0:
            # sensitivity filter (original code uses x * dc)
            dc[:] = np.asarray((H * (x * dc))[:, None].T / Hs)[:, 0] / np.maximum(
                0.001, x
            )
        elif ft == 1:
            # density filter
            dc[:] = np.asarray(H * (dc[:, None] / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[:, None] / Hs))[:, 0]

        # -------------------------
        # Update design variables
        # -------------------------
        if method.upper() == "BESO":
            # BESO update (simple version with fixed erase/add per iter)
            xold = x.copy()
            # number of solids currently
            idx_solid = np.where(x > 0.5)[0]
            n_solid = idx_solid.size

            if n_solid > n_target:
                # remove elements: among current solids, remove those with smallest ce
                n_remove = min(n_erase_per_iter, n_solid - n_target)
                # sort current solids by ce ascending
                order = np.argsort(ce[idx_solid])
                remove_idx = idx_solid[order[:n_remove]]
                x[remove_idx] = 0.0
            elif n_solid < n_target:
                # add elements: among holes, add those with largest ce
                idx_hole = np.where(x <= 0.5)[0]
                n_add = min(n_erase_per_iter, n_target - n_solid)
                order = np.argsort(-ce[idx_hole])
                add_idx = idx_hole[order[:n_add]]
                x[add_idx] = 1.0
            # Ensure we don't go below/above allowed [0,1]
            x = np.clip(x, 0.0, 1.0)
            xPhys[:] = x.copy()
            change = np.abs(x - xold).max()

        else:
            # Use OC or MMA as before for continuous methods (SIMP/RAMP)
            if xsolv == 0:
                xold1[:] = x
                (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)
            elif xsolv == 1:
                mu0 = 1.0  # Scale factor for objective function
                mu1 = 1.0  # Scale factor for volume constraint function
                f0val = mu0 * obj
                df0dx = mu0 * dc[:, None]
                fval = mu1 * np.array([[xPhys.sum() / (n * volfrac) - 1]])
                dfdx = mu1 * (dv / (n * volfrac))[None, :]
                xval = x.copy()[:, None]
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
                    m,
                    n,
                    k,
                    xval,
                    xmin,
                    xmax,
                    xold1,
                    xold2,
                    f0val,
                    df0dx,
                    fval,
                    dfdx,
                    low,
                    upp,
                    a0,
                    a,
                    c,
                    d,
                    move,
                )
                xold2 = xold1.copy()
                xold1 = xval.copy()
                x = xmma.copy().flatten()
            # Filter design variables
            if ft == 0:
                xPhys[:] = x
            elif ft == 1:
                xPhys[:] = np.asarray(H * x[:, None] / Hs)[:, 0]
            # Compute the change by the inf. norm / maximum change
            try:
                change = np.abs(x - xold1.flatten()).max()
            except:
                change = np.abs(x - xold1).max()

        # Print iteration
        print(
            "it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
                loop, obj, x.sum() / n, change
            )
        )

    # Plot result (same as original)
    fig, ax = plt.subplots()
    ax.imshow(
        -xPhys.reshape((nelx, nely)).T,
        cmap="gray",
        interpolation="none",
        norm=colors.Normalize(vmin=-1, vmax=0),
    )
    plt.show()

    # return useful results for post-processing if caller wants them
    return {"xPhys": xPhys.copy(), "obj": obj, "iter": loop, "change": change}
