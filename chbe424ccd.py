import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# Rxn 1 (Rxn 7 in the paper)
raw_C1_7 = 9.9068   # 10^-3 C1,i
raw_C2_7 = 7.4296   # 10^2 C2,i
raw_C3_7 = -5.3985  # 10^3 C3,i
raw_C4_7 = -20.2220 # C4,i

# Rxn 2 (Rxn 8 in the paper)
raw_C1_8 = 8.8226
raw_C2_8 = 0.8404
raw_C3_8 = 1.8736
raw_C4_8 = -21.6135

# Rxn 3 (Rxn 14 in the paper)
raw_C1_14 = -1.7352
raw_C2_14 = -4.7506
raw_C3_14 = 9.3576
raw_C4_14 = 5.6601

# adjusting values of C
def rescale_C(raw_C1, raw_C2, raw_C3, raw_C4):
    C1 = raw_C1 * 1e3 
    C2 = raw_C2 * 1e-2
    C3 = raw_C3 * 1e-3
    C4 = raw_C4
    return C1, C2, C3, C4

C1_1, C2_1, C3_1, C4_1 = rescale_C(raw_C1_7,  raw_C2_7,  raw_C3_7,  raw_C4_7)   # K1
C1_2, C2_2, C3_2, C4_2 = rescale_C(raw_C1_8,  raw_C2_8,  raw_C3_8,  raw_C4_8)   # K2
C1_3, C2_3, C3_3, C4_3 = rescale_C(raw_C1_14, raw_C2_14, raw_C3_14, raw_C4_14)  # K3

# equilibrium constants

def K1(T):
    # reaction 1
    return K_of_T(T, C1_1, C2_1, C3_1, C4_1)

def K2(T):
    # reaction 2
    return K_of_T(T, C1_2, C2_2, C3_2, C4_2)

def K3(T):
    # reaction 3
    return K_of_T(T, C1_3, C2_3, C3_3, C4_3)


# actually finding K
def lnK(T, C1, C2, C3, C4):
    return C1 / T + C2 * np.log(T) + C3 * T + C4

def K_of_T(T, C1, C2, C3, C4):
    return np.exp(lnK(T, C1, C2, C3, C4))

# evaluating over a temperature range [T1, T2]
T_min = 443.0 
T_max = 473.0
n_T = 7 # step size

T_values = np.linspace(T_min, T_max, n_T)

K1_values = K_of_T(T_values, C1_1, C2_1, C3_1, C4_1)
K2_values = K_of_T(T_values, C1_2, C2_2, C3_2, C4_2)
K3_values = K_of_T(T_values, C1_3, C2_3, C3_3, C4_3)



# list
species = [
    "H2O",       # 0
    "NH3",       # 1
    "CO2",       # 2
    "urea",      # 3
    "carbamate", # 4  (H2NCOO-)
    "bicarb",    # 5  (HCO3-)
    "NH4plus",   # 6  (NH4+)
]
ns = len(species)
idx = {s: i for i, s in enumerate(species)}

r = np.array([
    0.92,  # H2O
    1.00,  # NH3
    1.32,  # CO2
    2.16,  # urea (H2NCONH2)
    1.71,  # H2NCOO-
    1.54,  # HCO3-
    0.91,  # NH4+
])

q = np.array([
    1.40,  # H2O
    1.00,  # NH3
    1.12,  # CO2
    2.00,  # urea
    1.58,  # H2NCOO-
    1.44,  # HCO3-
    0.99,  # NH4+
])

# species charges
z = np.array([
    0,   # H2O
    0,   # NH3
    0,   # CO2
    0,   # urea
    -1,  # carbamate (H2NCOO-)
    -1,  # bicarb (HCO3-)
    +1,  # NH4+
])


Z_UNI = 10.0  # UNIQUAC coordination number

b_DH = 1.5  # distance-of-closest-approach parameter

# debye-huckel stuff

def A_phi_DH(T: float) -> float:
    """
    Debye-Huckel parameter. At 25 *C, it is
    0.509. Converted in terms of natural logs
    we get 1.17 via multiplying by ln(10)
    """
    return 0.509 * np.log(10)

def ionic_strength(x: np.ndarray) -> float:
    """
    I = 0.5 * sum(z_i^2 * m_i). 
    A reasonable approximation 
    is taking m_i ~ x_i.
    """
    return 0.5 * np.sum((z**2) * x)

def ln_gamma_long(T: float, x: np.ndarray) -> np.ndarray:
    """
    Debye–Huckel contribution ln(gamma_i^DH).
    Nonzero only for ionic species.
    """
    x = np.asarray(x, dtype=float)
    x_sum = x.sum()
    if x_sum <= 0:
        x_sum = 1e-16
    x = x / x_sum

    I = ionic_strength(x)
    sqrtI = np.sqrt(max(I, 1e-16))

    A_phi = A_phi_DH(T)
    ln_gamma_DH = np.zeros_like(x)

    if sqrtI > 0.0:
        for i in range(ns):
            if z[i] != 0:
                ln_gamma_DH[i] = -A_phi * z[i]**2 * sqrtI / (1.0 + b_DH * sqrtI)
            else:
                ln_gamma_DH[i] = 0.0
    return ln_gamma_DH



def tau_matrix(T: float) -> np.ndarray:
    a = np.zeros((ns, ns))

    # 1) NH3 – H2O
    i = idx["NH3"]; k = idx["H2O"]
    a_val = (4969.77 - 20.83235 * T + 0.0188211 * T**2)
    a[i, k] = a_val
    a_val = (-25642.10 + 107.7931 * T - 0.1086847 * T**2)
    a[k, i] = a_val

    # 2) CO2 – H2O
    i = idx["CO2"]; k = idx["H2O"]
    a_val = (-1272.667 + 183114.45 / T)
    a[i, k] = a_val
    a_val = (2282.919 - 334031.43 / T)
    a[k, i] = a_val

    # 3) NH4+ – H2O
    i = idx["NH4plus"]; k = idx["H2O"]
    a_val = -797.8
    a[i, k] = a_val
    a_val = 646.5
    a[k, i] = a_val

    # 4) HCO3- – H2O
    i = idx["bicarb"]; k = idx["H2O"]
    a_val = -772.5
    a[i, k] = a_val
    a_val = -474.4
    a[k, i] = a_val

    # 5) H2NCOO- – H2O
    i = idx["carbamate"]; k = idx["H2O"]
    a_val = -330.3
    a[i, k] = a_val
    a_val = 800.5
    a[k, i] = a_val

    # 6) NH3 – NH4+
    i = idx["NH3"]; k = idx["NH4plus"]
    a_val = 2500.0
    a[i, k] = a_val
    a_val = -154.0
    a[k, i] = a_val

    # 7) NH3 – carbamate
    i = idx["NH3"]; k = idx["carbamate"]
    a_val = 2500.0
    a[i, k] = a_val
    a_val = -657.0
    a[k, i] = a_val

    # 8) CO2 – NH4+
    i = idx["CO2"]; k = idx["NH4plus"]
    a_val = -634.0
    a[i, k] = a_val
    a_val = 1335.2
    a[k, i] = a_val

    # 9) CO2 – HCO3-
    i = idx["CO2"]; k = idx["bicarb"]
    a_val = -394.9
    a[i, k] = a_val
    a_val = -1061.5
    a[k, i] = a_val

    # 10) CO2 – carbamate
    i = idx["CO2"]; k = idx["carbamate"]
    a_val = -1026.5
    a[i, k] = a_val
    a_val = 217.7
    a[k, i] = a_val

    # 11) NH4+ – carbamate
    i = idx["NH4plus"]; k = idx["carbamate"]
    a_val = 2500.0
    a[i, k] = a_val
    a_val = -62.5
    a[k, i] = a_val

    # 12) NH4+ – HCO3-
    i = idx["NH4plus"]; k = idx["bicarb"]
    a_val = 1766.5
    a[i, k] = a_val
    a_val = 983.7
    a[k, i] = a_val

    tau = np.exp(-a / T)
    np.fill_diagonal(tau, 1.0)
    return tau

def ln_gamma_short(T: float, x: np.ndarray) -> np.ndarray:
    """
    Short-range UNIQUAC contribution:
      ln gamma_i^C  (combinatorial)
    + ln gamma_i^R  (residual)
    """
    x = np.asarray(x, dtype=float)
    x_sum = x.sum()
    if x_sum <= 0:
        x_sum = 1e-16
    x = x / x_sum
    x_safe = np.clip(x, 1e-16, 1.0)

    # volume and surface fractions
    phi = r * x_safe / np.dot(r, x_safe)
    theta = q * x_safe / np.dot(q, x_safe)

    # l_i
    l = (Z_UNI / 2.0) * (r - q) - (r - 1.0)

    # combinatorial term
    ln_gamma_C = (
        np.log(phi / x_safe)
        + (Z_UNI / 2.0) * q * np.log(theta / phi)
        + l
        - (phi / x_safe) * np.sum(x_safe * l)
    )

    # residual term
    tau = tau_matrix(T)
    theta_tau_col = theta @ tau 

    ln_gamma_R = np.zeros_like(x_safe)
    for i in range(ns):
        x1 = -np.log(theta_tau_col[i]) # term 1
        x2 = 0.0 # term 2, fortunately just zero for our case
        for j in range(ns):
            denom_j = np.dot(theta, tau[:, j])  # sum_k theta_k * tau_kj
            x2 += theta[j] * tau[i, j] / denom_j
        ln_gamma_R[i] = q[i] * (1.0 + x1 - x2)

    return ln_gamma_C + ln_gamma_R


def activity_coefficients(T: float, x: np.ndarray) -> np.ndarray:
    """
    gamma_i(T,x) = exp( ln gamma_i^C + ln gamma_i^R + ln gamma_i^DH )
    """
    ln_sr = ln_gamma_short(T, x)
    ln_dh = ln_gamma_long(T, x)
    ln_gamma = ln_sr + ln_dh
    return np.exp(ln_gamma)


# feed parameters
theta_CO2 = 1.0 # limiting reactant
theta_NH3 = 3.0 # tweak later so that e3/e2 is maximized. between 2.3 and 4
theta_H2O = 0.1 # arbitrary guess
theta_urea = 0.0 # assumption is zero urea in the fresh feed

# flow data for reactor
v0 = 1.0 # given
V = 1.0e4 # also given
F_CO2_0 = 1

# f_i = F_i/F_CO2
def f_eps(eps):
    eps1, eps2, eps3 = eps
    f = np.zeros(ns)
    f[idx["H2O"]] = theta_H2O - eps2 + eps3
    f[idx["NH3"]] = theta_NH3 - 2 * eps1 - eps2
    f[idx["CO2"]] = theta_CO2 - eps1 - eps2
    f[idx["urea"]] = theta_urea + eps3
    f[idx["carbamate"]] = eps1 - eps3
    f[idx["bicarb"]] = eps2
    f[idx["NH4plus"]] = eps1 + eps2 - eps3

    return f

def rxn_quotients(T, eps):

    f = f_eps(eps)

    # if any flow is <= 0
    if np.any(f <= 0.0):
        return None
    
    x = f / f.sum()
    gamma = activity_coefficients(T, x)
    a = gamma * x

    a_H2O    = a[idx["H2O"]]
    a_NH3    = a[idx["NH3"]]
    a_CO2    = a[idx["CO2"]]
    a_urea   = a[idx["urea"]]
    a_carb   = a[idx["carbamate"]]
    a_bicarb = a[idx["bicarb"]]
    a_NH4    = a[idx["NH4plus"]]

    Q1 = (a_carb * a_NH4) / (a_CO2 * a_NH3**2)
    Q2 = (a_bicarb * a_NH4) / (a_CO2 * a_NH3 * a_H2O)
    return Q1, Q2, a_carb, a_NH4, a_urea, a_H2O

# finding k3 with Arrhenius equation
def find_k3(T):
    A = 2.5e8 # in s^-1
    Ea = 100e3 # in J/mol
    R = 8.314 # in J/(mol K)
    return A * np.exp(-Ea / (R * T))

# residuals for the optimization
# TODO: do this later lmao
def residuals_eps(eps, T):
    # solving triple system of equations
    # 1. ln(Q1) - ln(K1) = 0
    # 2. ln(Q2) - ln(K2) = 0
    # 3. eps3 - r_urea*V/F_CO2_0 = 0

    out = rxn_quotients(T, eps)
    if out is None:
        return np.array([1e6, 1e6, 1e6])
    
    Q1, Q2, a_carb, a_NH4, a_urea, a_H2O = out

    eps_small = 1e-16
    Q1 = max(Q1, eps_small)
    Q2 = max(Q2, eps_small)

    res1 = np.log(Q1) - np.log(K1(T))
    res2 = np.log(Q2) - np.log(K2(T))

    k3 = find_k3(T)
    r_urea = k3 * a_carb * a_NH4

    # CSTR design equation
    res3 = eps[2] - r_urea * V

    return np.array([res1, res2, res3])

def solve_eps_CSTR(T):

    # initial guess for extents. tweak if needed
    eps0 = np.array([0.3, 0.1, 0.05])

    sol = root(residuals_eps, eps0, args=(T,))

    if not sol.success:
        raise RuntimeError(f"eps solve failed at T={T}: {sol.message}")

    eps_eq = sol.x
    return eps_eq

def scan_theta_NH3(T):
    global theta_NH3

    theta_vals = np.linspace(2.3, 4.0, 15)  # 15 points between 2.3 and 4
    best_theta = None
    best_metric = -np.inf
    best_eps = None

    for th in theta_vals:
        theta_NH3 = th  # update global
        try:
            eps_eq = solve_eps_CSTR(T)
        except RuntimeError:
            continue  # skip infeasible cases

        eps1, eps2, eps3 = eps_eq
        if eps2 <= 0:
            continue

        metric = eps3 / eps2   # e.g. "selectivity" proxy

        if metric > best_metric:
            best_metric = metric
            best_theta = th
            best_eps = eps_eq

    return best_theta, best_eps, best_metric

# actually running the code lmao
if __name__ == "__main__":

    # printing table of K1, K2, and K3
    print(" T [K]   K1 (carbamate)   K2 (bicarb)      K3 (urea)")
    for T, K1_val, K2_val, K3_val in zip(T_values, K1_values, K2_values, K3_values):
        print(f"{T:6.1f}  {K1_val:13.3e}  {K2_val:13.3e}  {K3_val:13.3e}")

    # adjust T as needed. Should be between 443 and 473 K
    T_test = 463.0 

    # feed composition is from guidelines
    # x_H2O = 0.2700, x_NH3 = 0.5338, x_CO2 = 0.0631, x_urea = 0.1331, others 0
    x_feed = np.zeros(ns)
    x_feed[idx["H2O"]] = 0.2700
    x_feed[idx["NH3"]] = 0.5338
    x_feed[idx["CO2"]] = 0.0631
    x_feed[idx["urea"]] = 0.1331

    # normalize in case they don't sum to exactly 1
    x_feed = x_feed / x_feed.sum()

    gamma_feed = activity_coefficients(T_test, x_feed)

    print(f"\nActivity coefficients at T = {T_test} K for feed composition:")
    for s, g in zip(species, gamma_feed):
        print(f"  {s:9s}: gamma = {g: .6f}")

    # solve for eps1, eps2, eps3 at T_test
    try:
        eps_eq = solve_eps_CSTR(T_test)
        print(f"\nCSTR extents at T = {T_test} K:")
        print(f"  eps1 (carbamate)   = {eps_eq[0]: .5e}")
        print(f"  eps2 (bicarb)      = {eps_eq[1]: .5e}")
        print(f"  eps3 (urea)        = {eps_eq[2]: .5e}")

        f_out = f_eps(eps_eq)
        f_tot = f_out.sum()
        print("\nOutlet composition (per 1 mol/s CO2 feed):")
        for s in species:
            fi = f_out[idx[s]]
            print(f"  {s:9s}: f = {fi: .5e}, x = {fi/f_tot: .5f}")
    except RuntimeError as e:
        print("\nCSTR solve failed:", e)

    best_theta, best_eps, best_metric = scan_theta_NH3(T_test)
    if best_theta is not None:
        print(f"\nBest theta_NH3 in [2.3,4] at T={T_test} K:")
        print(f"  theta_NH3* = {best_theta:.3f}, metric (eps3/eps2) = {best_metric:.3f}")
        print(f"  eps* = {best_eps}")
    else:
        print("\nNo feasible theta_NH3 found in [2.3,4].")