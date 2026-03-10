import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

from scipy.interpolate import splrep, splev
import sys, os
sys.path.insert(0, '/home/timm/Projects/PIML/subfoil')
from airfoil_utils import BSplineFoil
import vortexSourcePanelfoil_VecAnalytic as vortexSourcePanelfoil
import pyvista as pv

# ============================================================
# 1. AIRFOIL GEOMETRY (NACA 4-digit)
# ============================================================

def naca4_airfoil(naca='2412', n_points=101, te_closed=True):
    """
    Generate NACA 4-digit airfoil coordinates.
    Returns upper and lower surface points.
    """
    m = int(naca[0]) / 100.0   # max camber
    p = int(naca[1]) / 10.0    # max camber location
    t = int(naca[2:]) / 100.0  # thickness

    # Cosine spacing for better LE resolution
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    # Thickness distribution
    yt = 5 * t * (0.2969*np.sqrt(x)
                  - 0.1260*x
                  - 0.3516*x**2
                  + 0.2843*x**3
                  - 0.1015*x**4)

    if te_closed:
        yt[-1] = 0.0  # close trailing edge

    # Camber line
    yc = np.where(x < p,
                  m / p**2 * (2*p*x - x**2),
                  m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2))

    dyc_dx = np.where(x < p,
                      2*m / p**2 * (p - x),
                      2*m / (1-p)**2 * (p - x))

    theta = np.arctan(dyc_dx)

    # Upper and lower surfaces
    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl, x, yc


def geom_wake_x(x_start, ds_first, n_cells, grow, reverse=False):
    """Return n_cells+1 x-positions with exact geometric spacing.

    Cell widths: ds_first, ds_first*grow, ..., ds_first*grow^(n_cells-1).
    Wake domain length = ds_first * (grow^n_cells - 1) / (grow - 1).

    reverse=False → x_start → x_start+wake_len, small dx first (near x_start).
    reverse=True  → x_start+wake_len → x_start, small dx last  (near x_start = TE).
    """
    widths  = ds_first * grow ** np.arange(n_cells)   # small → large
    wake_len = widths.sum()
    if reverse:
        widths = widths[::-1]                          # large → small near TE
        offsets = np.concatenate([[0.0], np.cumsum(widths)])
        return (x_start + wake_len) - offsets          # x_TE+wake → x_TE
    offsets = np.concatenate([[0.0], np.cumsum(widths)])
    return x_start + offsets                           # x_TE → x_TE+wake


def cell_centers(X, Y):
    """Return cell-centre coordinates for a structured grid.

    For a node array of shape (n_eta, n_xi) the cell centres are the
    average of the 4 surrounding corner nodes, giving shape
    (n_eta-1, n_xi-1).  Using cell centres avoids sampling on or
    immediately adjacent to the airfoil surface.
    """
    Xc = 0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, :-1] + X[1:, 1:])
    Yc = 0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, :-1] + Y[1:, 1:])
    return Xc, Yc
# ============================================================

def generate_cmesh(naca='2412',
                   n_wrap=161,        # points wrapping around airfoil (ξ direction)
                   n_normal=81,       # points in normal direction (η direction)
                   r_far=15.0,        # far-field radius for the front arc (chord lengths)
                   growth=1.08,       # radial growth rate (η direction)
                   n_wake=15,          # wake columns per side (blocks A and C)
                   wake_growth=1.10):  # geometric growth rate of wake cells in x
                                       # wake_length = ds_TE*(wake_growth^n_wake-1)/(wake_growth-1)
    """
    Generate a structured C-mesh around an airfoil.

    C-mesh topology:
    - ξ direction: wraps around airfoil from TE → lower → LE → upper → TE
    - η direction: from airfoil surface to far field

    Outer boundary:
    - Front arc (LE region): outward normal extrusion to radius r_far
    - Wake cut (TE region):  straight +x extrusion, n_wake columns per side
    - Wake length is derived from the geometric series (ds_TE * growth series)

    Returns:
        X, Y    : mesh coordinates [n_normal, n_wrap]
        x_wall  : wall x-coordinates [n_wrap]
        y_wall  : wall y-coordinates [n_wrap]
        nx, ny  : outward unit normals at wall [n_wrap]
        r_dist  : radial distances from wall [n_normal]
    """
    print(f"Generating C-mesh: {n_wrap} x {n_normal} points")
    print(f"  Airfoil: NACA {naca}")
    print(f"  Far-field radius: {r_far} chords  (wake length derived from growth series)")

    x_TE = 1.0   # trailing edge chord position

    # ── Point counts ─────────────────────────────────────────────────
    n_lo  = int(n_wake)                        # columns per wake side (direct input)
    n_arc = n_wrap - 2 * n_lo                  # columns on airfoil
    n_whi = n_lo                               # (symmetric)

    if n_arc < 7:
        raise ValueError(
            f"n_wake={n_wake} leaves only n_arc={n_arc} columns for the foil "
            f"(n_wrap={n_wrap}).  Need n_arc ≥ 7, so n_wake ≤ {(n_wrap - 7) // 2}."
        )

    # ═══════════════════════════════════════════════════════════════
    #  THREE SEPARATE BLOCKS, TFI'd independently then concatenated
    # ═══════════════════════════════════════════════════════════════
    #
    #  BLOCK A  lower wake  (n_normal × n_lo)
    #   ξ: x_TE+wake_length → x_TE   (inner y=0, outer y=-r_far)
    #   Perfect rectangle: X constant per column, Y = ±s·r_far
    #
    #  BLOCK B  foil        (n_normal × n_arc)
    #   ξ: along airfoil surface
    #   η: foil surface → semicircular outer arc (centred at x_TE)
    #   Normal-direction TFI — airfoil shape is preserved exactly
    #
    #  BLOCK C  upper wake  (n_normal × n_whi)
    #   ξ: x_TE → x_TE+wake_length   (inner y=0, outer y=+r_far)
    #   Perfect rectangle
    # ═══════════════════════════════════════════════════════════════

    # --- η-distribution (shared by all blocks) ---
    ratios  = growth ** np.arange(n_normal)
    eta_raw = np.cumsum(ratios)
    s       = (eta_raw - eta_raw[0]) / (eta_raw[-1] - eta_raw[0])  # 0→1
    r_dist  = s * r_far

    # ── BLOCK A: lower wake (built after ds_lo is known) ────────────
    # x_wlo and X_A/Y_A are constructed below after foil surface is built

    # ── BLOCK B: foil ────────────────────────────────────────────────
    # Inner wall: airfoil surface  TE(lower) → LE → TE(upper)
    xu, yu, xl, yl, xc, yc = naca4_airfoil(naca, n_points=(n_arc + 1) // 2)
    x_foil_raw = np.concatenate([xl[::-1], xu[1:]])   # TE→LE→TE
    y_foil_raw = np.concatenate([yl[::-1], yu[1:]])
    t_raw      = np.linspace(0, 1, len(x_foil_raw))
    t_arc      = np.linspace(0, 1, n_arc)
    x_foil_w   = interp1d(t_raw, x_foil_raw, kind='cubic')(t_arc)
    y_foil_w   = interp1d(t_raw, y_foil_raw, kind='cubic')(t_arc)

    # TE arc-lengths: seed for geometric wake spacing
    ds_lo = float(np.hypot(x_foil_w[1]  - x_foil_w[0],  y_foil_w[1]  - y_foil_w[0]))
    ds_hi = float(np.hypot(x_foil_w[-1] - x_foil_w[-2], y_foil_w[-1] - y_foil_w[-2]))

    # Wake domain length derived from geometric series (no user input needed)
    wake_length = float(np.sum(ds_lo * wake_growth ** np.arange(n_lo)))

    # Outer arc: semicircle centred at (x_TE, 0), radius r_far
    # Angles: -π/2 (south, TE lower) → -3π/2 (north, TE upper)
    angles     = np.linspace(-np.pi / 2.0, -3.0 * np.pi / 2.0, n_arc)
    x_out_arc  = x_TE + r_far * np.cos(angles)
    y_out_arc  =        r_far * np.sin(angles)

    # TFI: each column interpolates foil→arc along η
    # X_B[k,j] = x_foil[j] + s[k]*(x_arc[j] - x_foil[j])
    X_B = x_foil_w + s[:, np.newaxis] * (x_out_arc - x_foil_w)   # (n_normal, n_arc)
    Y_B = y_foil_w + s[:, np.newaxis] * (y_out_arc - y_foil_w)

    # ── Geometric wake x-distributions (seeded by TE arc-lengths) ────
    x_wlo  = geom_wake_x(x_TE, ds_lo, n_lo,  wake_growth, reverse=True)
    X_A    = np.broadcast_to(x_wlo[np.newaxis, :],   (n_normal, n_lo + 1)).copy()
    Y_A    = np.broadcast_to(-s[:, np.newaxis] * r_far, (n_normal, n_lo + 1)).copy()

    # ── BLOCK C: upper wake ──────────────────────────────────────────
    # x from x_TE up to x_TE+wake_length  (leftmost col = TE junction)
    x_whi  = geom_wake_x(x_TE, ds_hi, n_whi, wake_growth, reverse=False)
    X_C    = np.broadcast_to(x_whi[np.newaxis, :],     (n_normal, n_whi + 1)).copy()
    Y_C    = np.broadcast_to(+s[:, np.newaxis] * r_far, (n_normal, n_whi + 1)).copy()

    # ── Concatenate blocks (drop shared TE boundary columns) ─────────
    # A[:,:-1]  → n_lo cols (x_TE+wake … just before x_TE)
    # X_B       → n_arc cols (x_TE lower TE … x_TE upper TE)
    # C[:,1:]   → n_whi cols (just after x_TE … x_TE+wake)
    # Total: n_lo + n_arc + n_whi = n_wrap  ✓
    X = np.concatenate([X_A[:, :-1], X_B, X_C[:, 1:]], axis=1)
    Y = np.concatenate([Y_A[:, :-1], Y_B, Y_C[:, 1:]], axis=1)
    assert X.shape == (n_normal, n_wrap), f"X shape mismatch {X.shape}"

    # ── Wall and normals for return ──────────────────────────────────
    x_wall = X[0, :]
    y_wall = Y[0, :]

    # Surface normals: wake zones point perp to wake cut; foil uses gradient
    nx = np.zeros(n_wrap)
    ny = np.zeros(n_wrap)
    ny[:n_lo]  = -1.0   # lower wake → points downward (outward from cut)
    ny[-n_whi:] = +1.0  # upper wake → points upward

    dx_f = np.gradient(x_foil_w)
    dy_f = np.gradient(y_foil_w)
    t_len = np.sqrt(dx_f**2 + dy_f**2) + 1e-14
    nxf = -dy_f / t_len
    nyf =  dx_f / t_len
    cx_f, cy_f = np.mean(x_foil_w), np.mean(y_foil_w)
    for i in range(n_arc):
        if np.dot([cx_f - x_foil_w[i], cy_f - y_foil_w[i]], [nxf[i], nyf[i]]) > 0:
            nxf[i] = -nxf[i]
            nyf[i] = -nyf[i]
    nx[n_lo:n_lo + n_arc] = nxf
    ny[n_lo:n_lo + n_arc] = nyf

    print(f"  Mesh generated: X shape = {X.shape}")
    print(f"  Block A (lower wake): {n_lo} cols, x=[{X_A[0,0]:.2f}..{X_A[0,-1]:.2f}], ds0={X_A[0,1]-X_A[0,0]:.4f}")
    print(f"  Block B (foil):       {n_arc} cols, LE≈({x_foil_w[n_arc//2]:.3f},{y_foil_w[n_arc//2]:.3f}), ds_TE_lo={ds_lo:.4f}, ds_TE_hi={ds_hi:.4f}")
    print(f"  Block C (upper wake): {n_whi} cols, x=[{X_C[0,0]:.2f}..{X_C[0,-1]:.2f}], ds0={X_C[0,1]-X_C[0,0]:.4f}")

    return X, Y, x_foil_w, y_foil_w, nx, ny, r_dist

# ============================================================
# 3. JACOBIAN COEFFICIENTS
# ============================================================

def compute_jacobian_coefficients(X, Y):
    """
    Compute the Jacobian and metric coefficients of the mapping
    (ξ, η) → (x, y).
    
    Metrics:
        x_xi  = ∂x/∂ξ ,  x_eta = ∂x/∂η
        y_xi  = ∂y/∂ξ ,  y_eta = ∂y/∂η
    
    Jacobian:
        J = x_xi * y_eta - x_eta * y_xi
    
    Inverse metrics (for PDE transformation):
        ξ_x  =  y_eta / J
        ξ_y  = -x_eta / J
        η_x  = -y_xi  / J
        η_y  =  x_xi  / J
    
    Geo-FNO input channels:
        [x, y, x_xi, x_eta, y_xi, y_eta, J, 
         g11, g12, g22,          # metric tensor
         xi_x, xi_y, eta_x, eta_y]  # inverse metrics
    """
    n_eta, n_xi = X.shape
    print(f"\nComputing Jacobian coefficients for {n_eta}×{n_xi} mesh...")

    # --- Metric derivatives (2nd-order central differences) ---
    x_xi  = np.zeros_like(X)
    x_eta = np.zeros_like(X)
    y_xi  = np.zeros_like(Y)
    y_eta = np.zeros_like(Y)

    # ∂/∂ξ  (axis=1, j-direction)
    x_xi[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2.0
    x_xi[:,    0] = (-3*X[:,0] + 4*X[:,1] - X[:,2]) / 2.0
    x_xi[:,   -1] = ( 3*X[:,-1] - 4*X[:,-2] + X[:,-3]) / 2.0

    y_xi[:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / 2.0
    y_xi[:,    0] = (-3*Y[:,0] + 4*Y[:,1] - Y[:,2]) / 2.0
    y_xi[:,   -1] = ( 3*Y[:,-1] - 4*Y[:,-2] + Y[:,-3]) / 2.0

    # ∂/∂η  (axis=0, k-direction)
    x_eta[1:-1, :] = (X[2:, :] - X[:-2, :]) / 2.0
    x_eta[   0, :] = (-3*X[0,:] + 4*X[1,:] - X[2,:]) / 2.0
    x_eta[  -1, :] = ( 3*X[-1,:] - 4*X[-2,:] + X[-3,:]) / 2.0

    y_eta[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / 2.0
    y_eta[   0, :] = (-3*Y[0,:] + 4*Y[1,:] - Y[2,:]) / 2.0
    y_eta[  -1, :] = ( 3*Y[-1,:] - 4*Y[-2,:] + Y[-3,:]) / 2.0

    # --- Jacobian ---
    J = x_xi * y_eta - x_eta * y_xi

    # --- Metric tensor components ---
    # g_ij = (∂r/∂ξ^i) · (∂r/∂ξ^j)
    g11 = x_xi**2  + y_xi**2    # |∂r/∂ξ|²
    g12 = x_xi*x_eta + y_xi*y_eta  # ∂r/∂ξ · ∂r/∂η
    g22 = x_eta**2 + y_eta**2   # |∂r/∂η|²

    # --- Inverse metrics ---
    J_safe = np.where(np.abs(J) < 1e-12, 1e-12, J)
    xi_x  =  y_eta / J_safe
    xi_y  = -x_eta / J_safe
    eta_x = -y_xi  / J_safe
    eta_y =  x_xi  / J_safe

    # --- PDE coefficients for Laplace in curvilinear coords ---
    # ∇²φ = 0 transforms to:
    # α φ_ξξ - 2β φ_ξη + γ φ_ηη + lower-order = 0
    # where:
    alpha = g22 / J_safe**2  # coefficient of φ_ξξ  (= |∇η|²)
    beta  = g12 / J_safe**2  # coefficient of φ_ξη
    gamma = g11 / J_safe**2  # coefficient of φ_ηη  (= |∇ξ|²)

    # Quality metrics
    J_min, J_max = J.min(), J.max()
    J_neg = np.sum(J < 0)
    print(f"  Jacobian range: [{J_min:.4f}, {J_max:.4f}]")
    print(f"  Negative Jacobians: {J_neg} cells")
    print(f"  Grid aspect ratio (max): "
          f"{np.max(np.sqrt(g22)/np.sqrt(g11+1e-14)):.2f}")

    coeffs = {
        'x': X, 'y': Y,
        'x_xi': x_xi, 'x_eta': x_eta,
        'y_xi': y_xi, 'y_eta': y_eta,
        'J': J,
        'g11': g11, 'g12': g12, 'g22': g22,
        'xi_x': xi_x, 'xi_y': xi_y,
        'eta_x': eta_x, 'eta_y': eta_y,
        'alpha': alpha, 'beta': beta, 'gamma': gamma
    }
    return coeffs

def run_panel_method(x_wall, y_wall, X, Y,
                     alpha_aoa=-5.0, U_inf=1.0):

    x_ccw = x_wall[-1:0:-1]
    y_ccw = y_wall[-1:0:-1]

    panel_foil = vortexSourcePanelfoil.PanelFoil(x_ccw, y_ccw, U_inf, alpha_aoa)
    #panel_foil.plot_panels()
    #panel_foil.debug_normals()

    panel_foil.solve_vortex_and_source_strengths()
    panel_foil.compute_tangential_velocities()
    panel_foil.compute_pressure_coefficients()
    panel_foil.plot_pressure_coefficients(path='./potential/pressure_coefficients.png')
    panel_foil.check_solution()
    panel_foil.compute_lift_coefficient()
    panel_foil.compute_coefficients_from_pressure()

    # Use cell-centre coordinates: centres are half a cell away from the wall
    # so k=0 centres are never on a panel and near-singular behaviour is avoided.
    X_c, Y_c = cell_centers(X, Y)   # shape (n_eta-1, n_xi-1)

    U_out, V_out, *_ = panel_foil.compute_velocity_field(X_c, Y_c)
    U  = U_out
    V  = V_out
    Cp = 1.0 - (U_out**2 + V_out**2) / U_inf**2

    # define a mesh grid
    nx, ny = 100, 20  # number of points in the x and y directions
    x_start, x_end = -1.0, 2.0
    y_start, y_end = -0.3, 0.3
    Xg, Yg = np.meshgrid(np.linspace(x_start, x_end, nx), np.linspace(y_start, y_end, ny))

    panel_foil.plot_velocity_field(Xg, Yg, path = './potential/Vec_Analytic_velocity_field.png')
    panel_foil.plot_pressure_field(Xg, Yg, path = './potential/Vec_Analytic_pressure_field.png')

    return X_c, Y_c, U, V, Cp


# ============================================================
# 4. POTENTIAL FLOW SOLVER
# ============================================================

def potential_flow_cmesh(X, Y, coeffs, 
                          alpha_aoa=5.0,  # angle of attack (degrees)
                          U_inf=1.0):     # freestream velocity
    """
    Solve Laplace equation ∇²φ = 0 on C-mesh using finite differences
    in curvilinear coordinates.
    
    PDE in (ξ,η) space:
        α φ_ξξ - 2β φ_ξη + γ φ_ηη 
        + [∂α/∂ξ + ∂β/∂η... ] φ_ξ + [...] φ_η = 0
    
    Boundary conditions:
        - Far field: φ = U∞(x cosα + y sinα)  [Dirichlet]
        - Wall: ∂φ/∂n = 0  [Neumann, flow tangency]
        - Wake cut: periodic/symmetric
    
    Returns velocity potential φ and velocities (u, v).
    """
    aoa_rad = np.radians(alpha_aoa)
    n_eta, n_xi = X.shape
    print(f"\nSolving potential flow...")
    print(f"  AoA = {alpha_aoa}°,  U∞ = {U_inf}")
    print(f"  Grid: {n_eta} × {n_xi} = {n_eta*n_xi} DOFs")

    alpha_c = coeffs['alpha']   # φ_ξξ coefficient
    beta_c  = coeffs['beta']    # φ_ξη coefficient
    gamma_c = coeffs['gamma']   # φ_ηη coefficient

    N = n_eta * n_xi
    def idx(k, j):
        return k * n_xi + j

    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N)

    # Far-field potential
    def phi_ff(x, y):
        return U_inf * (x * np.cos(aoa_rad) + y * np.sin(aoa_rad))

    print("  Assembling linear system...")
    
    for k in range(n_eta):
        for j in range(n_xi):
            row = idx(k, j)

            # ── Far-field boundary (η = n_eta-1) ──────────────────
            if k == n_eta - 1:
                A[row, row] = 1.0
                b[row] = phi_ff(X[k,j], Y[k,j])
                continue

            # ── Wall boundary (η = 0): ∂φ/∂η = 0 (Neumann) ───────
            if k == 0:
                # One-sided difference: φ[1,j] - φ[0,j] = 0
                A[row, idx(0, j)] = -1.0
                A[row, idx(1, j)] =  1.0
                b[row] = 0.0
                continue

            # ── Wake cut (ξ = 0 and ξ = n_xi-1): periodic ────────
            if j == 0:
                # Periodic: φ[k,0] = φ[k,-1]
                A[row, idx(k, 0)]      =  1.0
                A[row, idx(k, n_xi-1)] = -1.0
                b[row] = 0.0
                continue

            if j == n_xi - 1:
                A[row, row] = 1.0
                A[row, idx(k, n_xi-2)] = -1.0
                b[row] = 0.0
                continue

            # ── Interior: curvilinear Laplace ─────────────────────
            a  = alpha_c[k, j]
            be = beta_c[k, j]
            ga = gamma_c[k, j]

            # φ_ξξ  ≈ φ[k,j+1] - 2φ[k,j] + φ[k,j-1]
            # φ_ηη  ≈ φ[k+1,j] - 2φ[k,j] + φ[k-1,j]
            # φ_ξη  ≈ (φ[k+1,j+1] - φ[k+1,j-1] - φ[k-1,j+1] + φ[k-1,j-1]) / 4

            A[row, idx(k,   j+1)] += a
            A[row, idx(k,   j-1)] += a
            A[row, idx(k,   j  )] -= 2*a

            A[row, idx(k+1, j  )] += ga
            A[row, idx(k-1, j  )] += ga
            A[row, idx(k,   j  )] -= 2*ga

            A[row, idx(k+1, j+1)] -= 0.5 * be
            A[row, idx(k+1, j-1)] += 0.5 * be
            A[row, idx(k-1, j+1)] += 0.5 * be
            A[row, idx(k-1, j-1)] -= 0.5 * be

    print("  Solving sparse system...")
    A_csr = A.tocsr()
    phi = spsolve(A_csr, b)
    phi = phi.reshape(n_eta, n_xi)

    # ── Velocities from potential ─────────────────────────────────
    xi_x  = coeffs['xi_x']
    xi_y  = coeffs['xi_y']
    eta_x = coeffs['eta_x']
    eta_y = coeffs['eta_y']

    # ∂φ/∂ξ
    phi_xi = np.zeros_like(phi)
    phi_xi[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / 2.0
    phi_xi[:,    0] = phi[:, 1] - phi[:, 0]
    phi_xi[:,   -1] = phi[:, -1] - phi[:, -2]

    # ∂φ/∂η
    phi_eta = np.zeros_like(phi)
    phi_eta[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / 2.0
    phi_eta[   0, :] = phi[1, :] - phi[0, :]
    phi_eta[  -1, :] = phi[-1, :] - phi[-2, :]

    # u = ∂φ/∂x = φ_ξ·ξ_x + φ_η·η_x
    u = phi_xi * xi_x + phi_eta * eta_x
    v = phi_xi * xi_y + phi_eta * eta_y

    speed = np.sqrt(u**2 + v**2)
    print(f"  Velocity range: [{speed.min():.3f}, {speed.max():.3f}]")
    print(f"  Cp range: [{(1-speed**2).min():.3f}, {(1-speed**2).max():.3f}]")

    return phi, u, v

# ============================================================
# 5. RUN EVERYTHING
# ============================================================

# Parameters
NACA     = '2212'
N_WRAP   = 201    # reduced for speed; use 161+ for production
N_NORMAL = 61
R_FAR    = 3.0
AOA      = 10.0
N_WAKE   = 50    # wake columns per side; must satisfy n_wake <= (N_WRAP-7)//2
#WAKE_FRAC    = 0.15   # fraction of wrap used for wake cut per side
WAKE_GROWTH  = 1.1   # geometric growth rate of wake cells in x
                      # wake_length = ds_TE*(WAKE_GROWTH^n_wake - 1)/(WAKE_GROWTH - 1)


# Generate mesh
X, Y, x_wall, y_wall, nx_wall, ny_wall, r_dist = \
    generate_cmesh(NACA, N_WRAP, N_NORMAL, R_FAR,
                   n_wake=N_WAKE, wake_growth=WAKE_GROWTH)

# Compute Jacobian coefficients
jacob_coeffs = compute_jacobian_coefficients(X, Y)

# Solve potential flow
X_c, Y_c, u, v, Cp = run_panel_method(x_wall,y_wall, X, Y,  alpha_aoa=AOA, U_inf=1.0)


def plot_physical_space(X,Y,u,v,Cp,coeffs):
    n_eta, n_xi = X.shape

    # Build a flat 3-D point array (z = 0)
    Z = np.zeros_like(X)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # PyVista StructuredGrid expects (n_xi, n_eta, n_z) order
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (n_xi, n_eta, 1)
    x_xi = coeffs['x_xi']
    x_eta = coeffs['x_eta']
    y_xi = coeffs['y_xi']
    y_eta = coeffs['y_eta']
    J = coeffs['J']


    data_map = {
            'x': X, 'y': Y, 'u': u, 'v': v, 'Cp': Cp, 
            'J': J, 'x_xi': x_xi, 'x_eta': x_eta, 
            'y_xi': y_xi, 'y_eta': y_eta
        }

    for field_name, field in data_map.items():
        if field.shape == (n_eta - 1, n_xi - 1):
            print(f"Adding Cell data to Physical space for {field_name}...{field.shape}")

            grid.cell_data[field_name] = field.ravel(order='C')
        else:
            print(f"Adding Point data to Physical space for {field_name}...{field.shape}")
            grid.point_data[field_name] = field.ravel(order='C')

    pl = pv.Plotter(shape=(1, 3), off_screen=False)
    # Cp Plot
    pl.subplot(0, 0)
    pl.add_mesh(grid.copy(), scalars="Cp", cmap="RdBu_r", clim=[-2.0, 1.0])
    pl.add_text("Cp (Physical Space)")
    pl.view_xy()

    # U Velocity Plot
    pl.subplot(0, 1)
    pl.add_mesh(grid.copy(), scalars="u", cmap="RdBu_r", clim=[-2.0, 2.0])
    pl.add_text("U Velocity (Physical Space)")
    pl.view_xy()

    # V Velocity Plot
    pl.subplot(0, 2)
    pl.add_mesh(grid.copy(), scalars="v", cmap="RdBu_r", clim=[-2.0, 2.0])
    pl.add_text("V Velocity (Physical Space)")
    pl.view_xy()    

    pl.show()
    pl.screenshot(f"physical/naca_{NACA}_{AOA}_physical_space_{field_name}.png", window_size=[1024, 768])
    pl.close()
    
    # Second Plotter for Jacobian Coefficients
    pl2 = pv.Plotter(shape=(2, 3))
    
    plots = [
        (0, 0, "x_xi", "x_xi"),
        (0, 1, "x_eta", "x_eta"),
        (1, 0, "y_xi", "y_xi"),
        (1, 1, "y_eta", "y_eta"),
        (1, 2, "J", "Jacobian J")
    ]
    
    for r, c, field, label in plots:
        pl2.subplot(r, c)
        pl2.add_mesh(grid.copy(), scalars=field, cmap="RdBu_r",show_edges=False, edge_color='lightgray', line_width=0.4)
        pl2.add_text(label)
        pl2.view_xy()
    pl2.show()
    pl2.screenshot(f"physical/naca_{NACA}_{AOA}_meshcoeffs.png", window_size=[1024, 768])

    pl2.close()      



def plot_latent_space(X,Y,u,v,Cp,coeffs):
    n_eta, n_xi = X.shape
    xi_coords = np.arange(n_xi)
    eta_coords = np.arange(n_eta)
    XI, ETA, Z = np.meshgrid(xi_coords, eta_coords, [0], indexing='ij')

    latent_grid = pv.StructuredGrid(XI, ETA, Z)
    J = coeffs['J']
    x_xi = coeffs['x_xi']
    x_eta = coeffs['x_eta']
    y_xi = coeffs['y_xi']
    y_eta = coeffs['y_eta']
    data_map = {
            'x': X, 'y': Y, 'u': u, 'v': v, 'Cp': Cp, 
            'J': J, 'x_xi': x_xi, 'x_eta': x_eta, 
            'y_xi': y_xi, 'y_eta': y_eta
        }

    for field_name, field in data_map.items():
        print(f"Plotting latent space for {field_name}...{field.shape}")
        if field.shape == (n_eta - 1, n_xi - 1):
            latent_grid.cell_data[field_name] = field.ravel(order='C')
        else:
            latent_grid.point_data[field_name] = field.ravel(order='C')

    pl = pv.Plotter(shape=(1, 3), off_screen=False)
    # Cp Plot
    pl.subplot(0, 0)
    pl.add_mesh(latent_grid.copy(), scalars="Cp", cmap="RdBu_r", clim=[-2.0, 1.0])
    pl.add_text("Cp (Latent Space)")
    pl.view_xy()

    # U Velocity Plot
    pl.subplot(0, 1)
    pl.add_mesh(latent_grid.copy(), scalars="u", cmap="RdBu_r", clim=[-2.0, 2.0])
    pl.add_text("U Velocity (Latent Space)")
    pl.view_xy()

    # V Velocity Plot
    pl.subplot(0, 2)
    pl.add_mesh(latent_grid.copy(), scalars="v", cmap="RdBu_r",clim=[-2.0, 2.0])
    pl.add_text("V Velocity (Latent Space)")
    pl.view_xy()    

    pl.show()
    pl.screenshot(f"latent/naca_{NACA}_{AOA}_latent_space_{field_name}.png", window_size=[1024, 768])
    pl.close()
    
    # Second Plotter for Jacobian Coefficients
    pl2 = pv.Plotter(shape=(2, 3))
    
    plots = [
        (0, 0, "x_xi", "x_xi"),
        (0, 1, "x_eta", "x_eta"),
        (1, 0, "y_xi", "y_xi"),
        (1, 1, "y_eta", "y_eta"),
        (1, 2, "J", "Jacobian J")
    ]
    
    for r, c, field, label in plots:
        pl2.subplot(r, c)
        pl2.add_mesh(latent_grid.copy(), scalars=field, cmap="RdBu_r",show_edges=True, edge_color='lightgray', line_width=0.4)
        pl2.add_text(label)
        pl2.view_xy()
    pl2.show()
    pl2.screenshot(f"latent/naca_{NACA}_{AOA}_meshcoeffs.png", window_size=[1024, 768])

    pl2.close()      
#==========================================================
# 6. PYVISTA MESH VISUALISATION
# ============================================================

def plot_cmesh_pyvista(X, Y, field=None, field_name='Cp',
                       title='C-Mesh', show_edges=True,
                       edge_skip=1, n_lo=None, n_arc=None):
    """
    Render the structured C-mesh in an interactive PyVista window.

    Parameters
    ----------
    X, Y       : mesh coordinates [n_normal, n_wrap]
    field      : scalar field to colour the mesh (same shape as X)
    field_name : label used in the colour bar
    title      : window / plotter title
    show_edges : draw mesh edges (may be slow for large grids)
    edge_skip  : plot only every Nth mesh line as an overlay
    """
    n_eta, n_xi = X.shape

    # Build a flat 3-D point array (z = 0)
    Z = np.zeros_like(X)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # PyVista StructuredGrid expects (n_xi, n_eta, n_z) order
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (n_xi, n_eta, 1)

    # Attach scalar field — cell-centred or node data detected automatically
    if field is not None:
        if field.shape == (n_eta - 1, n_xi - 1):
            grid.cell_data[field_name] = field.ravel(order='C')
        else:
            grid.point_data[field_name] = field.ravel(order='C')

    # ── Separate wall geometry: foil (smooth spline) + wake cuts (straight) ──
    # If n_lo/n_arc not supplied, fall back to drawing the whole wall as a line.
    if n_lo is not None and n_arc is not None:
        # Foil section: X[0, n_lo : n_lo+n_arc]  — high-res spline
        foil_pts = np.column_stack([
            X[0, n_lo:n_lo + n_arc],
            Y[0, n_lo:n_lo + n_arc],
            np.zeros(n_arc)
        ])
        wall_line = pv.Spline(foil_pts, n_arc * 8)   # 8× oversample for smooth curve

        # Lower wake cut: X[0, 0:n_lo+1]  — straight horizontal line at y≈0
        lo_pts = np.column_stack([
            X[0, :n_lo + 1], Y[0, :n_lo + 1], np.zeros(n_lo + 1)
        ])
        lo_line = pv.lines_from_points(lo_pts)

        # Upper wake cut: X[0, n_lo+n_arc-1:]  — straight horizontal line at y≈0
        hi_pts = np.column_stack([
            X[0, n_lo + n_arc - 1:],
            Y[0, n_lo + n_arc - 1:],
            np.zeros(X.shape[1] - (n_lo + n_arc - 1))
        ])
        hi_line = pv.lines_from_points(hi_pts)
    else:
        # Fallback: whole wall as a polyline (no spline distortion at corners)
        wall_pts = np.column_stack([X[0, :], Y[0, :], np.zeros(X.shape[1])])
        wall_line = pv.lines_from_points(wall_pts)
        lo_line = hi_line = None

    # ── Build plotter ─────────────────────────────────────────
    pl = pv.Plotter(title=title)
    pl.set_background('white')

    if field is not None:
        pl.add_mesh(grid,
                    scalars=field_name,
                    cmap='RdBu_r',
                    show_edges=show_edges,
                    edge_color='lightgray',
                    line_width=0.4,
                    opacity=1.0,
                    scalar_bar_args=dict(title=field_name, vertical=True))
    else:
        pl.add_mesh(grid,
                    color='lightblue',
                    show_edges=show_edges,
                    edge_color='steelblue',
                    line_width=0.4)

    # Highlight airfoil wall and wake cuts
    pl.add_mesh(wall_line, color='black', line_width=3, label='Airfoil')
    if lo_line is not None:
        pl.add_mesh(lo_line, color='gray', line_width=2)
        pl.add_mesh(hi_line, color='gray', line_width=2)

    # Overlay every edge_skip-th η-line for readability
    for k in range(0, n_eta, edge_skip):
        line_pts = np.column_stack([X[k, :], Y[k, :], np.zeros(X[k, :].shape)])
        pl.add_mesh(pv.Spline(line_pts, n_xi),
                    color='dimgray', line_width=0.6, opacity=0.5)

    pl.add_axes()
    pl.view_xy()
    pl.show()
    pl.screenshot(f"potential/c_mesh_NACA_{NACA}_{AOA}_{field_name}.png", window_size=[1920, 1080])
    pl.close()



# Plot mesh coloured by Cp
_n_lo  = N_WAKE
_n_arc = N_WRAP - 2 * N_WAKE


x_xi = jacob_coeffs['x_xi']
x_eta = jacob_coeffs['x_eta']
y_xi = jacob_coeffs['y_xi']
y_eta = jacob_coeffs['y_eta']
J = jacob_coeffs['J']

U_mag = np.sqrt(u**2 + v**2)

plot_cmesh_pyvista(X, Y, field=Cp, field_name='Cp',show_edges=False,
                    title=f'C-Mesh  NACA {NACA}  AoA={AOA}°',
                    n_lo=_n_lo, n_arc=_n_arc)

plot_cmesh_pyvista(X, Y, field=U_mag, field_name='U_mag',show_edges=False,
                   title=f'C-Mesh  NACA {NACA}  AoA={AOA}°',
                   n_lo=_n_lo, n_arc=_n_arc)

plot_cmesh_pyvista(X, Y, field=J, field_name='J',
                    title=f'C-Mesh  NACA {NACA}  AoA={AOA}°',
                    n_lo=_n_lo, n_arc=_n_arc)

plot_cmesh_pyvista(X, Y, field=x_xi, field_name='x_xi',
                    title=f'C-Mesh  NACA {NACA}  AoA={AOA}°',
                    n_lo=_n_lo, n_arc=_n_arc)

plot_cmesh_pyvista(X, Y, field=x_eta, field_name='x_eta',
                    title=f'C-Mesh  NACA {NACA}  AoA={AOA}°',
                    n_lo=_n_lo, n_arc=_n_arc)

plot_cmesh_pyvista(X, Y, field=y_xi, field_name='y_xi',
                    title=f'C-Mesh  NACA {NACA}  AoA={AOA}°',
                   n_lo=_n_lo, n_arc=_n_arc)

plot_cmesh_pyvista(X, Y, field=y_eta, field_name='y_eta',
                   title=f'C-Mesh  NACA {NACA}  AoA={AOA}°',
                   n_lo=_n_lo, n_arc=_n_arc)

plot_latent_space(X,Y,u,v,Cp,jacob_coeffs)


plot_physical_space(X,Y,u,v,Cp,jacob_coeffs)


