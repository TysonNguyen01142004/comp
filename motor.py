from method import main, evaluate_layout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# ---------- Helper function for consistent density plotting ----------
def save_density(x_vec, nelx, nely, title, fname):
    """Plot density with fixed normalization: 0=white (void), 1=black (solid)."""
    arr = x_vec.reshape((nelx, nely)).T
    plt.figure(figsize=(6, 3))
    plt.imshow(
        arr,
        cmap="gray_r",  # white = void, black = solid
        interpolation="none",
        aspect="auto",
        norm=colors.Normalize(vmin=0.0, vmax=1.0),  # fixed scale for all
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=220)
    print(f"Saved {fname}")
    # plt.show()  # Uncomment if you want to display each figure interactively


# ---------- Run settings ----------
case = "armor"  # our realistic motorcycle armor case
nelx = 80
nely = 40
vol = 0.35
penal = 3.0
rmin = 2.0
ft = 1  # density filter
xsolv = 0  # OC optimizer

# ---------- 1) Run topology optimization ----------
res = main(
    nelx=nelx,
    nely=nely,
    volfrac=vol,
    penal=penal,
    rmin=rmin,
    ft=ft,
    xsolv=xsolv,
    method="SIMP",
    case=case,
    max_iter=150,
    tol=1e-3,
)

x_opt = res["xPhys"]  # optimized density field

# ---------- 2) Build conventional layouts ----------
n = nelx * nely
x_uniform = vol * np.ones(n)  # uniform thin panel
x_solid = np.ones(n)  # 100% solid (for reference)

# X-brace panel: diagonal ribs, same volume fraction
x_xbrace = np.zeros(n)
thick = max(2, int(0.12 * min(nelx, nely)))  # ~12% thickness
grid = x_xbrace.reshape(nelx, nely)
for i in range(nelx):
    j1 = int(round((nely - 1) * i / (nelx - 1)))
    j2 = int(round((nely - 1) * (nelx - 1 - i) / (nelx - 1)))
    jlo1 = max(0, j1 - thick)
    jhi1 = min(nely - 1, j1 + thick)
    jlo2 = max(0, j2 - thick)
    jhi2 = min(nely - 1, j2 + thick)
    grid[i, jlo1 : jhi1 + 1] = 1.0
    grid[i, jlo2 : jhi2 + 1] = 1.0
scale = vol / (grid.mean() + 1e-12)
x_xbrace = np.clip(grid * scale, 0.0, 1.0).reshape(-1)


# ---------- 3) Evaluate layouts ----------
obj_opt, uy_opt = evaluate_layout(nelx, nely, penal, rmin, ft, case, x_opt)
obj_uni, uy_uni = evaluate_layout(nelx, nely, penal, rmin, ft, case, x_uniform)
obj_x, uy_x = evaluate_layout(nelx, nely, penal, rmin, ft, case, x_xbrace)
obj_solid, uy_solid = evaluate_layout(nelx, nely, penal, rmin, ft, case, x_solid)


# ---------- 4) Print comparison table ----------
print("\n=== Comparison @ equal mass (volfrac = {:.2f}) ===".format(vol))
print("{:<14} {:>14} {:>14}".format("Design", "Compliance C", "Center uy"))
print("-" * 46)
print("{:<14} {:>14.6e} {:>14.6e}".format("Optimized", obj_opt, uy_opt))
print("{:<14} {:>14.6e} {:>14.6e}".format("Uniform", obj_uni, uy_uni))
print("{:<14} {:>14.6e} {:>14.6e}".format("X-brace", obj_x, uy_x))
print("\n(reference with different mass)")
print("{:<14} {:>14.6e} {:>14.6e}".format("Solid(100%)", obj_solid, uy_solid))


# ---------- 5) Save result figures ----------
save_density(x_opt, nelx, nely, f"Optimized (vol={vol:.2f})", "armor_optimized.png")
save_density(x_uniform, nelx, nely, "Uniform panel (same mass)", "armor_uniform.png")
save_density(x_xbrace, nelx, nely, "X-brace (same mass)", "armor_xbrace.png")

print("\n✅ All results saved! Check:")
print(" - armor_optimized.png")
print(" - armor_uniform.png")
print(" - armor_xbrace.png")
