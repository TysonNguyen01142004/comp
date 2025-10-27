# method.py
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
from matplotlib import colors
import matplotlib.pyplot as plt

# mmapy imports left as-is for MMA option
try:
    from mmapy import mmasub, subsolv
except Exception:
    # If mmapy not available, MMA option will raise if used.
    mmasub = None
    subsolv = None


# element stiffness matrix (unchanged)
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


# Optimality criterion (unchanged)
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0
    l2 = 1e9
    move = 0.2
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


# Modified main: added `case` plus small performance/safety tweaks
def main(
    nelx,
    nely,
    volfrac,
    penal,
    rmin,
    ft,
    xsolv,
    method="SIMP",
    ramp_q=0.5,
    b_erase=0.02,
    case="benchmark",
    max_iter=150,
    tol=1e-3,
):
    """
    main(..., case='benchmark'|'armor'|'cantilever')
    - case controls load/support pattern for different realistic setups.
    - max_iter: iteration cap for speed.
    - tol: convergence tolerance on density change.
    """
    print("Topology optimization - method:", method)
    print("ndes: " + str(nelx) + " x " + str(nely))
    print(
        "volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal)
    )
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    print("Optimizer: " + ["OC method", "MMA"][xsolv])
    print("Case:", case)

    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0

    # dofs:
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Allocate design variables (as array), initialize and allocate sens.
    n = nely * nelx

    # initial designs:
    if method.upper() == "BESO":
        x = np.ones(n, dtype=float)
    else:
        x = volfrac * np.ones(n, dtype=float)

    xPhys = x.copy()
    dc = np.zeros((nely, nelx), dtype=float).flatten()

    # Initialize OC
    if xsolv == 0:
        xold1 = x.copy()
        g = 0
    elif xsolv == 1:
        # prepare MMA arrays — will error if mmapy not installed
        if mmasub is None:
            raise RuntimeError("mmapy not available: cannot use MMA optimizer")
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
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # BC's and support — choose based on `case`
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))

    if case == "cantilever":
        # classic cantilever: left edge fixed
        fixed = np.union1d(
            dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1])
        )
    elif case == "armor" or case == "armor_plate" or case == "motorcycle_armor":
        # armor plate: fix left and right mounting strips (simulate attachments) to reduce rigid body modes
        left_fix = np.arange(0, 2 * (nely + 1), 1)  # left vertical edge (both dofs)
        right_fix = np.arange(
            2 * nelx * (nely + 1), 2 * (nelx + 1) * (nely + 1), 1
        )  # right vertical edge
        # also optionally fix bottom corners to prevent rigid motion
        bottom_right = np.array([2 * (nelx + 1) * (nely + 1) - 1])
        fixed = np.union1d(left_fix, right_fix)
        fixed = np.union1d(fixed, bottom_right)
    else:
        # default fallback: left edge fixed (safe)
        fixed = np.union1d(
            dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1])
        )

    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load depending on case
    if case == "cantilever":
        # downward point load at free end (typical benchmark)
        f[1, 0] = -1
    elif case == "armor" or case == "armor_plate" or case == "motorcycle_armor":
        # apply a concentrated load (impact) near center-top region
        # choose node approximately at (nelx*0.5, nely*0.75) to simulate upper impact
        ix = int(round(nelx * 0.5))
        iy = int(round(nely * 0.75))
        node = ix * (nely + 1) + iy
        f[2 * node + 1, 0] = -1.0
    else:
        # default
        f[1, 0] = -1

    # Set loop counter and gradient vectors
    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    # For BESO control
    if method.upper() == "BESO":
        n_target = int(np.round(volfrac * n))
        n_erase_per_iter = max(1, int(np.round(b_erase * n)))

    # Main loop
    while (change > tol) and (loop < max_iter):
        loop = loop + 1

        # FE: build stiffness matrix
        if method.upper() == "SIMP":
            E_elems = Emin + xPhys**penal * (Emax - Emin)
        elif method.upper() == "RAMP":
            q = ramp_q
            denom = 1.0 + q * (1.0 - xPhys)
            E_elems = Emin + (xPhys / denom) * (Emax - Emin)
        elif method.upper() == "BESO":
            E_elems = Emin + xPhys * (Emax - Emin)
        else:
            raise ValueError("Unknown method: " + str(method))

        sK = (KE.flatten()[:, None] * (E_elems)).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]

        # Use conjugate gradient for speed; fallback to spsolve if it fails
        try:
            u_free, info = cg(K, f[free, 0], maxiter=500, tol=1e-6)
            if info != 0:
                # CG did not converge; fallback to direct solve
                u[free, 0] = spsolve(K, f[free, 0])
            else:
                u[free, 0] = u_free
        except Exception:
            u[free, 0] = spsolve(K, f[free, 0])

        # Objective and sensitivities
        ce[:] = (
            np.dot(u[edofMat].reshape(nelx * nely, 8), KE)
            * u[edofMat].reshape(nelx * nely, 8)
        ).sum(1)
        obj = (E_elems * ce).sum()

        if method.upper() == "SIMP":
            dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        elif method.upper() == "RAMP":
            q = ramp_q
            dE_dx = (Emax - Emin) * (1.0 + q) / (1.0 + q * (1.0 - xPhys)) ** 2
            dc[:] = -dE_dx * ce
        elif method.upper() == "BESO":
            dc[:] = -(Emax - Emin) * ce

        dv[:] = np.ones(nely * nelx)

        # Filtering
        if ft == 0:
            dc[:] = np.asarray((H * (x * dc))[:, None] / Hs)[:, 0] / np.maximum(
                0.001, x
            )
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[:, None] / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[:, None] / Hs))[:, 0]

        # Update
        if method.upper() == "BESO":
            xold = x.copy()
            idx_solid = np.where(x > 0.5)[0]
            n_solid = idx_solid.size

            if n_solid > n_target:
                n_remove = min(n_erase_per_iter, n_solid - n_target)
                order = np.argsort(ce[idx_solid])
                remove_idx = idx_solid[order[:n_remove]]
                x[remove_idx] = 0.0
            elif n_solid < n_target:
                idx_hole = np.where(x <= 0.5)[0]
                n_add = min(n_erase_per_iter, n_target - n_solid)
                order = np.argsort(-ce[idx_hole])
                add_idx = idx_hole[order[:n_add]]
                x[add_idx] = 1.0
            x = np.clip(x, 0.0, 1.0)
            xPhys[:] = x.copy()
            change = np.abs(x - xold).max()
        else:
            if xsolv == 0:
                xold1[:] = x
                (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)
            elif xsolv == 1:
                m = 1
                mu0 = 1.0
                mu1 = 1.0
                f0val = mu0 * obj
                df0dx = mu0 * dc[:, None]
                fval = mu1 * np.array([[xPhys.sum() / (n * volfrac) - 1]])
                dfdx = mu1 * (dv / (n * volfrac))[None, :]
                xval = x.copy()[:, None]
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
                    m,
                    n,
                    0,
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
            if ft == 0:
                xPhys[:] = x
            elif ft == 1:
                xPhys[:] = np.asarray(H * x[:, None] / Hs)[:, 0]
            try:
                change = np.abs(x - xold1.flatten()).max()
            except:
                change = np.abs(x - xold1).max()

        # Print iteration
        print(
            "it.: {0} , obj.: {1:.6f} Vol.: {2:.3f}, ch.: {3:.6f}".format(
                loop, obj, x.sum() / n, change
            )
        )

    # Plot result
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(
        -xPhys.reshape((nelx, nely)).T,
        cmap="gray",
        interpolation="none",
        norm=colors.Normalize(vmin=-1, vmax=0),
        aspect="auto",
    )
    ax.set_title(f"Result: {case} | method={method} | vol={volfrac}")
    plt.axis("off")
    plt.show()

    return {"xPhys": xPhys.copy(), "obj": obj, "iter": loop, "change": change}


# ---------- Utility: evaluate a fixed layout without optimizing ----------
def evaluate_layout(nelx, nely, penal, rmin, ft, case, xPhys_in):
    """
    Compute compliance and center displacement for a given density field xPhys_in (no updates).
    Uses the same BCs and loading as main(..., case=...).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import cg, spsolve

    KE = lk()
    ndof = 2 * (nelx + 1) * (nely + 1)
    n = nelx * nely

    # --- Build filter (needed for density filter consistency) ---
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
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # edof
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
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # BCs & load like in main(case)
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    if case in ["armor", "armor_plate", "motorcycle_armor"]:
        left_fix = np.arange(0, 2 * (nely + 1), 1)
        right_fix = np.arange(2 * nelx * (nely + 1), 2 * (nelx + 1) * (nely + 1), 1)
        bottom_right = np.array([2 * (nelx + 1) * (nely + 1) - 1])
        fixed = np.union1d(np.union1d(left_fix, right_fix), bottom_right)
        ix = int(round(nelx * 0.5))
        iy = int(round(nely * 0.75))
        load_node = ix * (nely + 1) + iy
    else:
        fixed = np.union1d(
            dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1])
        )
        load_node = 0  # classic demo uses f[1] below

    free = np.setdiff1d(dofs, fixed)
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    if case in ["armor", "armor_plate", "motorcycle_armor"]:
        f[2 * load_node + 1, 0] = -1.0
    else:
        f[1, 0] = -1.0

    # If using density filter, filter xPhys_in the same way
    xPhys = xPhys_in.copy()
    if ft == 1:
        xPhys = np.asarray(H * xPhys[:, None] / Hs)[:, 0]

    Emin, Emax = 1e-9, 1.0
    E_elems = Emin + (xPhys**penal) * (Emax - Emin)

    sK = (KE.flatten()[:, None] * (E_elems)).flatten(order="F")
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K = K[free, :][:, free]

    try:
        u_free, info = cg(K, f[free, 0], tol=1e-6, maxiter=500)
        if info != 0:
            from scipy.sparse.linalg import spsolve

            u[free, 0] = spsolve(K, f[free, 0])
        else:
            u[free, 0] = u_free
    except Exception:
        from scipy.sparse.linalg import spsolve

        u[free, 0] = spsolve(K, f[free, 0])

    ce = (
        np.dot(u[edofMat].reshape(nelx * nely, 8), KE)
        * u[edofMat].reshape(nelx * nely, 8)
    ).sum(1)
    obj = (E_elems * ce).sum()

    # report the vertical displacement at load node (if used)
    uy_center = (
        u[2 * load_node + 1, 0]
        if case in ["armor", "armor_plate", "motorcycle_armor"]
        else u[1, 0]
    )

    return float(obj), float(uy_center)
