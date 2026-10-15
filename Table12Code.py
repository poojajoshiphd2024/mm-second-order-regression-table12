import numpy as np
from scipy.optimize import curve_fit

#  ---------Build the generator matrix  ----------
def build_generator(b, c, N, lambd, mu):
    Nq = N - b * c
    def iA(j, r):
        return r * b + j            # Region A: r=0..c-1, j=0..b-1

    def iB(j):
        return b * c + j            # Region B: r=c, j=0..Nq

    n = b * c + Nq + 1
    Q = np.zeros((n, n))
    e = np.zeros(n)

    def rowsA(r):
        return slice(r * b, r * b + b)

    colsB = slice(b * c, n)          # size Nq+1

    # ---------- Block B0 ----------
    B0 = np.zeros((b, b))
    for i in range(b):
        B0[i, i] = -lambd
        if i < b - 1:
            B0[i, i + 1] = lambd
    Q[rowsA(0), rowsA(0)] = B0

    # ---------- Blocks Br, r = 1..c-1 -------
    for r in range(1, c):
        Br = np.zeros((b, b))
        for i in range(b):
            Br[i, i] = -(lambd + r * mu)
            if i < b - 1:
                Br[i, i + 1] = lambd
        Q[rowsA(r), rowsA(r)] = Br

    # ---------- Blocks Ar, r = 0..c-2 -------------
    for r in range(0, c - 1):
        Ar = np.zeros((b, b))
        Ar[b - 1, 0] = lambd
        Q[rowsA(r), rowsA(r + 1)] = Ar

    # ---------- Boundary block Ac-1 ---------------
    Ac1 = np.zeros((b, Nq + 1))
    Ac1[b - 1, 0] = lambd
    Q[rowsA(c - 1), colsB] = Ac1

    # ---------- Blocks Cr, r = 1..c-1 ----------
    for r in range(1, c):
        Cr = r * mu * np.eye(b)
        Q[rowsA(r), rowsA(r - 1)] = Cr

    # ---------- Block Cc ----------
    Cc = np.zeros((Nq + 1, b))
    for j in range(min(b, Nq + 1)):
        Cc[j, j] = c * mu
    Q[colsB, rowsA(c - 1)] = Cc

    # ---------- Block Bc ----------
    Bc = np.zeros((Nq + 1, Nq + 1))
    for j in range(Nq + 1):
        if j < Nq:
            Bc[j, j] = -(lambd + c * mu)
            Bc[j, j + 1] = lambd
        else:
            Bc[j, j] = -c * mu
        if j >= b:
            Bc[j, j - b] = c * mu
    Q[colsB, colsB] = Bc

    # ---------- State labels ----------
    state_labels = [None] * n
    for r in range(c):
        for j in range(b):
            state_labels[iA(j, r)] = (j, r)
    for j in range(Nq + 1):
        state_labels[iB(j)] = (j, c)

    return Q, e, state_labels, n


# ------------ LU decomposition -------------
def lu_decompose(A):
    n = A.shape[0]
    A = A.copy().astype(float)
    L = np.eye(n)
    U = np.zeros((n, n))

    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])

        if abs(U[k, k]) < 1e-14:
            raise ZeroDivisionError("diagonal entry zero")

        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U


def forward_substitution(L, e):
    """Solve L y = e (L unit lower-triangular)."""
    n = len(e)
    y = np.zeros(n)
    for i in range(n):
        y[i] = e[i] - np.dot(L[i, :i], y[:i])
    return y


def back_substitution(U, y):
    """Solve U x = y (U upper-triangular)."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


#--------- Full solve for given b, c, N, lambda, mu-----------
def solve_queue_lu(b, c, N, lambd, mu):
    """Solves the CTMC stationary distribution and returns mean number in system L."""
    if lambd == 0:
        return 0.0

    Q, e, state_labels, n = build_generator(b, c, N, lambd, mu)

    QT = Q.T.copy()
    QT[-1, :] = 1.0
    e[-1] = 1.0

    L_mat, U_mat = lu_decompose(QT)

    y = forward_substitution(L_mat, e)
    x = back_substitution(U_mat, y)

    x = np.clip(x, 0, None)
    x /= x.sum()

    L_value = sum((j + r * b) * x[idx] for idx, (j, r) in enumerate(state_labels))
    return L_value

# -------Michaelis-Menten type model----------
def michaelis_menten_model(rho_val, Vmax, Km):
    return (Vmax * rho_val**2) / (Km + rho_val**2)

# -------------Main----------------------
if __name__ == "__main__":
    c, N, mu = 4, 30, 8
    b = 7  
    print(f"M/M^b/c/N  b={b}  c={c}  N={N}  mu={mu}\n")
    print(f"{'rho':>6}  {'lambda':>8}  {'L':>10}")
    print("-" * 30)

    # Step 1: rho and lambda arrays
    rho_values = np.arange(0.1, 5.1, 0.4)
    lambd_values = rho_values * b * c * mu

    # Step 2: Calculate L from balance equations
    L_values = []
    for rho_i, lambd_i in zip(rho_values, lambd_values):
        L = solve_queue_lu(b, c, N, lambd_i, mu)
        L_values.append(L)
        print(f"{rho_i:6.2f}  {lambd_i:8.2f}  {L:10.2f}")
    analytical_values = np.array(L_values)

    # Step 3: Fitting Michaelis-Menten model for analytical values
    p0 = [analytical_values.max() * 2, 1]
    params, _ = curve_fit(michaelis_menten_model, rho_values, analytical_values,
                          p0=p0)
    Vmax_fit, Km_fit = params
    mm_fit_values = michaelis_menten_model(rho_values, Vmax_fit, Km_fit)

    # Step 4: goodness of fit 
    residuals = analytical_values - mm_fit_values
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((analytical_values - np.mean(analytical_values)) ** 2)
    R_squared = (1 - (ss_res / ss_tot)) * 100

    print(f"\nBest-fit parameters: Vmax = {Vmax_fit:.2f}, Km = {Km_fit:.2f}")
    print(f"R-squared: {R_squared:.1f}")

    # Step 5: comparison table (analytical vs MM fit, for the base b)
    print(f"\n{'rho':>6}  {'Analytical L':>14}  {'MM fit L':>10}  {'abs diff':>10}")
    print("-" * 46)
    for r, a, m in zip(rho_values, analytical_values, mm_fit_values):
        print(f"{r:6.2f}  {a:14.4f}  {m:10.4f}  {abs(a-m):10.4f}")

    # Step 6: 
    b_list = [4, 5, 6, 7]
    L_analytical_by_b = {}
    L_mm_by_b = {}
    mm_params_by_b = {}
    R2_by_b = {}
    lambd_by_b = {}

    for b_val in b_list:
        lambd_values_b = rho_values * b_val * c * mu  
        lambd_by_b[b_val] = lambd_values_b
        L_this_b = []
        for lambd_i in lambd_values_b:
            L_this_b.append(solve_queue_lu(b_val, c, N, lambd_i, mu))
        L_this_b = np.array(L_this_b)
        L_analytical_by_b[b_val] = L_this_b

        p0_b = [L_this_b.max() * 2, 1.0]
        params_b, _ = curve_fit(michaelis_menten_model, rho_values, L_this_b,
                                p0=p0_b)
        mm_params_by_b[b_val] = params_b
        L_mm_by_b[b_val] = michaelis_menten_model(rho_values, *params_b)

        res_b = L_this_b - L_mm_by_b[b_val]
        R2_by_b[b_val] = (1 - np.sum(res_b ** 2) /
                          np.sum((L_this_b - np.mean(L_this_b)) ** 2)) * 100

    print(f"\nMichaelis-Menten fit parameters by b (c={c}, N={N}, mu={mu}):")
    for b_val in b_list:
        Vmax_b, Km_b = mm_params_by_b[b_val]
        print(f"  b={b_val:>3}:  Vmax = {Vmax_b:7.4f}   Km = {Km_b:7.4f}   "
              f"R-squared = {R2_by_b[b_val]:6.2f}")

    
    header = f"{'rho':>6}" + "".join(
        f"  {('lambda(b='+str(bv)+')'):>13}  {('L(b='+str(bv)+')'):>10}" for bv in b_list
    )
    print(f"\nlambda and MM-model L for fixed c={c}, N={N}, mu={mu}, across b = {b_list}\n")
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(rho_values):
        row = f"{r:6.2f}" + "".join(
            f"  {lambd_by_b[bv][idx]:13.2f}  {L_mm_by_b[bv][idx]:10.4f}" for bv in b_list
        )
        print(row)