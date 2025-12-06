import numpy as np
from scipy.optimize import root

# finding equilibrium constants at some T

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

def rescale_C(raw_C1, raw_C2, raw_C3, raw_C4):
    C1 = raw_C1 * 1e3 
    C2 = raw_C2 * 1e-2
    C3 = raw_C3 * 1e-3
    C4 = raw_C4
    return C1, C2, C3, C4

C1_1, C2_1, C3_1, C4_1 = rescale_C(raw_C1_7,  raw_C2_7,  raw_C3_7,  raw_C4_7)   # K1
C1_2, C2_2, C3_2, C4_2 = rescale_C(raw_C1_8,  raw_C2_8,  raw_C3_8,  raw_C4_8)   # K2
C1_3, C2_3, C3_3, C4_3 = rescale_C(raw_C1_14, raw_C2_14, raw_C3_14, raw_C4_14)  # K3

def lnK(T, C1, C2, C3, C4):
    return C1 / T + C2 * np.log(T) + C3 * T + C4

def K_of_T(T, C1, C2, C3, C4):
    return np.exp(lnK(T, C1, C2, C3, C4))

def K1(T):
    return K_of_T(T, C1_1, C2_1, C3_1, C4_1)

def K2(T):
    return K_of_T(T, C1_2, C2_2, C3_2, C4_2)

def K3(T):
    return K_of_T(T, C1_3, C2_3, C3_3, C4_3)

# species and stoichiometry

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

nu = np.zeros((ns, 3))
nu[idx["H2O"],      :] = [0.0, -1.0, +1.0]
nu[idx["NH3"],      :] = [-2.0, -1.0, 0.0]
nu[idx["CO2"],      :] = [-1.0, -1.0, 0.0]
nu[idx["urea"],     :] = [0.0, 0.0, +1.0]
nu[idx["carbamate"],:] = [+1.0, 0.0, -1.0]
nu[idx["bicarb"],   :] = [0.0, +1.0, 0.0]
nu[idx["NH4plus"],  :] = [+1.0, +1.0, -1.0]

# UNIQUAC size/shape parameters
r = np.array([
    0.92,  # H2O
    1.00,  # NH3
    1.32,  # CO2
    2.16,  # urea
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
z_charges = np.array([
    0,   # H2O
    0,   # NH3
    0,   # CO2
    0,   # urea
    -1,  # carbamate
    -1,  # bicarb
    +1,  # NH4+
])

Z_UNI = 10.0
b_DH  = 1.5

# long-range Debye–Hückel contribution 
def A_phi_DH(T: float) -> float:
    return 0.509 * np.log(10.0)

def ionic_strength(x: np.ndarray) -> float:
    return 0.5 * np.sum((z_charges**2) * x)

def ln_gamma_long(T: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_sum = x.sum()
    if x_sum <= 0.0:
        x_sum = 1e-16
    x = x / x_sum

    I = ionic_strength(x)
    sqrtI = np.sqrt(max(I, 1e-16))

    A_phi = A_phi_DH(T)
    ln_gamma_DH = np.zeros_like(x)

    if sqrtI > 0.0:
        for i in range(ns):
            if z_charges[i] != 0:
                ln_gamma_DH[i] = -A_phi * z_charges[i]**2 * sqrtI / (1.0 + b_DH * sqrtI)
    return ln_gamma_DH

# uniquac short-range contribution (combinatorial + residual terms)

def tau_matrix(T: float) -> np.ndarray:
    a = np.zeros((ns, ns))

    # 1) NH3 – H2O
    i = idx["NH3"]; k = idx["H2O"]
    a[i, k] = (4969.77 - 20.83235 * T + 0.0188211 * T**2)
    a[k, i] = (-25642.10 + 107.7931 * T - 0.1086847 * T**2)

    # 2) CO2 – H2O
    i = idx["CO2"]; k = idx["H2O"]
    a[i, k] = (-1272.667 + 183114.45 / T)
    a[k, i] = (2282.919 - 334031.43 / T)

    # 3) NH4+ – H2O
    i = idx["NH4plus"]; k = idx["H2O"]
    a[i, k] = -797.8
    a[k, i] = 646.5

    # 4) HCO3- – H2O
    i = idx["bicarb"]; k = idx["H2O"]
    a[i, k] = -772.5
    a[k, i] = -474.4

    # 5) H2NCOO- – H2O
    i = idx["carbamate"]; k = idx["H2O"]
    a[i, k] = -330.3
    a[k, i] = 800.5

    # 6) NH3 – NH4+
    i = idx["NH3"]; k = idx["NH4plus"]
    a[i, k] = 2500.0
    a[k, i] = -154.0

    # 7) NH3 – carbamate
    i = idx["NH3"]; k = idx["carbamate"]
    a[i, k] = 2500.0
    a[k, i] = -657.0

    # 8) CO2 – NH4+
    i = idx["CO2"]; k = idx["NH4plus"]
    a[i, k] = -634.0
    a[k, i] = 1335.2

    # 9) CO2 – HCO3-
    i = idx["CO2"]; k = idx["bicarb"]
    a[i, k] = -394.9
    a[k, i] = -1061.5

    # 10) CO2 – carbamate
    i = idx["CO2"]; k = idx["carbamate"]
    a[i, k] = -1026.5
    a[k, i] = 217.7

    # 11) NH4+ – carbamate
    i = idx["NH4plus"]; k = idx["carbamate"]
    a[i, k] = 2500.0
    a[k, i] = -62.5

    # 12) NH4+ – HCO3-
    i = idx["NH4plus"]; k = idx["bicarb"]
    a[i, k] = 1766.5
    a[k, i] = 983.7

    tau = np.exp(-a / T)
    np.fill_diagonal(tau, 1.0)
    return tau

def ln_gamma_short_return_parts(T: float, x: np.ndarray):
    """
    Return (ln_gamma_C, ln_gamma_R) for UNIQUAC (symmetric reference).
    """
    x = np.asarray(x, dtype=float)
    x_sum = x.sum()
    if x_sum <= 0.0:
        x_sum = 1e-16
    x = x / x_sum
    x_safe = np.clip(x, 1e-16, 1.0)

    phi   = r * x_safe / np.dot(r, x_safe)
    theta = q * x_safe / np.dot(q, x_safe)

    l = (Z_UNI / 2.0) * (r - q) - (r - 1.0)

    ln_gamma_C = (
        np.log(phi / x_safe)
        + (Z_UNI / 2.0) * q * np.log(theta / phi)
        + l
        - (phi / x_safe) * np.sum(x_safe * l)
    )

    tau = tau_matrix(T)
    theta_tau_col = theta @ tau

    ln_gamma_R = np.zeros_like(x_safe)
    for i in range(ns):
        x1 = -np.log(theta_tau_col[i])
        x2 = 0.0
        for j in range(ns):
            denom_j = np.dot(theta, tau[:, j])
            x2 += theta[j] * tau[i, j] / denom_j
        ln_gamma_R[i] = q[i] * (1.0 + x1 - x2)

    return ln_gamma_C, ln_gamma_R

def ln_gamma_short(T: float, x: np.ndarray) -> np.ndarray:
    lnC, lnR = ln_gamma_short_return_parts(T, x)
    return lnC + lnR

def activity_coefficients(T: float, x: np.ndarray) -> np.ndarray:
    """
    Symmetric activity coefficients including Debye–Hückel:
        gamma_i = exp( ln gamma_i^C + ln gamma_i^R + ln gamma_i^DH ).
    """
    ln_sr = ln_gamma_short(T, x)
    ln_dh = ln_gamma_long(T, x)
    return np.exp(ln_sr + ln_dh)

def activity_coefficients_unsymmetric(T: float, x: np.ndarray) -> np.ndarray:
    """
    Unsymmetric activity coefficients (reference = pure water).
    Convert symmetric UNIQUAC gammas to unsymmetric standard state
    and then add the same Debye–Hückel contribution.
    """
    x = np.asarray(x, dtype=float)
    x_sum = x.sum()
    if x_sum <= 0.0:
        x_sum = 1e-16
    x = x / x_sum

    lnC, lnR = ln_gamma_short_return_parts(T, x)
    ln_sym_short = lnC + lnR

    ref = idx["H2O"]
    l_vec = (Z_UNI / 2.0) * (r - q) - (r - 1.0)

    ln_gamma_inf_C = np.zeros(ns)
    ln_gamma_inf_R = np.zeros(ns)

    tau = tau_matrix(T)
    for i in range(ns):
        ln_gamma_inf_C[i] = (
            np.log(r[i] / r[ref])
            + (Z_UNI / 2.0) * q[i] * np.log((q[i] * r[ref]) / (q[ref] * r[i]))
            + l_vec[i]
            - r[i] * l_vec[ref] / q[ref]
        )
        tau_ref_i = tau[ref, i]
        ln_gamma_inf_R[i] = -q[i] * (np.log(tau_ref_i) - 1.0 + tau_ref_i)

    ln_unsym_short = ln_sym_short - ln_gamma_inf_C - ln_gamma_inf_R
    ln_unsym_short[ref] = 0.0  # gamma°_H2O = 1 by definition

    ln_dh = ln_gamma_long(T, x)
    return np.exp(ln_unsym_short + ln_dh)

# reactor setup

theta_CO2 = 1.0
theta_NH3 = 4.0
theta_H2O = 0.1
theta_urea = 0.0

v0 = 1000.0
V  = 1.00e5  # total reactor volume (arbitrary units)

def F_in_from_theta(theta_NH3_value=None):
    """
    Inlet molar flows based on the stoichiometric ratios.
    Default uses the global theta_NH3.
    """
    if theta_NH3_value is None:
        theta_NH3_value = theta_NH3

    F_in = np.zeros(ns)
    F_in[idx["H2O"]]  = theta_H2O
    F_in[idx["NH3"]]  = theta_NH3_value
    F_in[idx["CO2"]]  = theta_CO2
    F_in[idx["urea"]] = theta_urea
    return F_in

def f_eps_stage(eps, F_in):
    return F_in + nu @ eps

def rxn_quotients(T, F_out):
    if np.any(F_out <= 0.0):
        return None

    x = F_out / F_out.sum()
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

def find_k3(T):
    A  = 2.5e8   # s^-1
    Ea = 100e3   # J/mol
    R  = 8.314   # J/(mol K)
    return A * np.exp(-Ea / (R * T))

def residuals_eps_stage(eps, T, F_in, V_stage):
    F_out = f_eps_stage(eps, F_in)

    out = rxn_quotients(T, F_out)
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

    res3 = eps[2] - r_urea * V_stage

    return np.array([res1, res2, res3])

def solve_one_CSTR(T, F_in, V_stage):
    eps0 = np.array([0.1, 0.05, 0.01])
    sol = root(residuals_eps_stage, eps0, args=(T, F_in, V_stage))

    if not sol.success:
        raise RuntimeError(f"CSTR solve failed: {sol.message}")

    eps = sol.x
    F_out = f_eps_stage(eps, F_in)
    return eps, F_out

def run_CSTR_series(T, N_stages, V_total):
    V_stage = V_total / N_stages
    F_in = F_in_from_theta()

    eps_list = []
    for _ in range(N_stages):
        eps_k, F_out = solve_one_CSTR(T, F_in, V_stage)
        eps_list.append(eps_k)
        F_in = F_out

    eps_array = np.vstack(eps_list)
    F_final = F_in
    return eps_array, F_final

def scan_theta_NH3(T, V_total):
    theta_vals = np.linspace(2.3, 4.0, 15)
    best_theta = None
    best_metric = -np.inf
    best_eps = None

    for th in theta_vals:
        F_in = F_in_from_theta(theta_NH3_value=th)
        try:
            eps_eq, _ = solve_one_CSTR(T, F_in, V_total)
        except RuntimeError:
            continue

        eps1, eps2, eps3 = eps_eq
        if eps2 <= 0.0:
            continue

        metric = eps3 / eps2
        if metric > best_metric:
            best_metric = metric
            best_theta = th
            best_eps = eps_eq

    return best_theta, best_eps, best_metric

# actually running stuff (the main block)

if __name__ == "__main__":
    # K1–K3 table
    T_min = 443.0 
    T_max = 473.0
    n_T   = 7
    T_values = np.linspace(T_min, T_max, n_T)
    K1_values = K_of_T(T_values, C1_1, C2_1, C3_1, C4_1)
    K2_values = K_of_T(T_values, C1_2, C2_2, C3_2, C4_2)
    K3_values = K_of_T(T_values, C1_3, C2_3, C3_3, C4_3)

    print(" T [K]   K1 (carbamate)   K2 (bicarb)      K3 (urea)")
    for T, K1_val, K2_val, K3_val in zip(T_values, K1_values, K2_values, K3_values):
        print(f"{T:6.1f}  {K1_val:13.3e}  {K2_val:13.3e}  {K3_val:13.3e}")

    # Design temperature
    T_test = 463.0
    V_total = V

    # Feed composition from guidelines (for gamma table only)
    x_feed = np.zeros(ns)
    x_feed[idx["H2O"]]  = 0.2700
    x_feed[idx["NH3"]]  = 0.5338
    x_feed[idx["CO2"]]  = 0.0631
    x_feed[idx["urea"]] = 0.1331
    x_feed = x_feed / x_feed.sum()

    gamma_sym = activity_coefficients(T_test, x_feed)
    gamma_unsym = activity_coefficients_unsymmetric(T_test, x_feed)

    print(f"\nSymmetric activity coefficients γ_i at T = {T_test} K (feed composition):")
    for s, g in zip(species, gamma_sym):
        print(f"  {s:9s}: gamma = {g: .6f}")

    print(f"\nUnsymmetric activity coefficients γ_i° (water reference) at T = {T_test} K:")
    for s, g in zip(species, gamma_unsym):
        print(f"  {s:9s}: gamma° = {g: .6f}")

    # Single CSTR with total volume V_total, using theta ratios
    try:
        F_in0 = F_in_from_theta()
        eps_eq, f_out = solve_one_CSTR(T_test, F_in0, V_total)

        print(f"\nSingle CSTR (V = {V_total:.2e} L) at T = {T_test} K (basis: 1 mol/s CO2):")
        print(f"  eps1 (carbamate)   = {eps_eq[0]: .5e}")
        print(f"  eps2 (bicarb)      = {eps_eq[1]: .5e}")
        print(f"  eps3 (urea)        = {eps_eq[2]: .5e}")

        # overall reaction selectivity
        S_urea_bicarb = eps_eq[2] / eps_eq[1]
        print(f"  Selectivity S_urea/bicarb = eps3/eps2 = {S_urea_bicarb:.3f}")

        f_tot = f_out.sum()
        print("\nOutlet composition for basis F_CO2,in = 1 mol/s:")
        for s in species:
            fi = f_out[idx[s]]
            print(f"  {s:9s}: F = {fi: .5e}, x = {fi/f_tot: .5f}")

        # Urea production for basis 1 mol/s CO2
        MW_UREA = 60.06  # g/mol
        prod_kg_per_day_basis = eps_eq[2] * MW_UREA / 1000.0 * 86400.0
        print(f"\nUrea production for F_CO2,in = 1 mol/s: {prod_kg_per_day_basis:.1f} kg/day")

        # Target: 1000 kg/day of urea
        target_kg_day = 1000.0
        target_mol_s = target_kg_day * 1000.0 / (MW_UREA * 86400.0)

        eps3_per_CO2 = eps_eq[2]  # mol urea / s per 1 mol/s CO2
        F_CO2_needed = target_mol_s / eps3_per_CO2

        print(f"\nTo produce {target_kg_day:.0f} kg/day of urea (same T, V, and ratios):")
        print(f"  Required CO2 feed  F_CO2,in ≈ {F_CO2_needed:.3f} mol/s")
        print(f"                      (≈ {F_CO2_needed * 3600/1000:.3f} kmol/h)")

        F_NH3_needed = theta_NH3 * F_CO2_needed / theta_CO2
        F_H2O_needed = theta_H2O * F_CO2_needed / theta_CO2

        print(f"  Corresponding NH3 feed      ≈ {F_NH3_needed:.3f} mol/s")
        print(f"  Corresponding H2O feed      ≈ {F_H2O_needed:.3f} mol/s")

    except RuntimeError as e:
        print("\nSingle CSTR solve failed:", e)

    # Optional: CSTRs in series
    for N in [1, 2, 3, 50]:
        try:
            eps_array, F_final = run_CSTR_series(T_test, N, V_total)
        except RuntimeError as e:
            print(f"\nN = {N}: solve failed – {e}")
            continue

        eps_total = eps_array.sum(axis=0)
        x_final = F_final / F_final.sum()

        S_total = eps_total[2] / eps_total[1]

        print(f"\n=== {N} CSTR(s) in series at T = {T_test} K ===")
        print(f"Total eps1 (carbamate) = {eps_total[0]:.4e}")
        print(f"Total eps2 (bicarb)    = {eps_total[1]:.4e}")
        print(f"Total eps3 (urea)      = {eps_total[2]:.4e}")
        print(f"x_urea at outlet       = {x_final[idx['urea']]:.5f}")
        print(f"Overall S_urea/bicarb  = {S_total:.3f}")
    
        # Scan theta_NH3 to maximize selectivity eps3/eps2
    best_theta, best_eps, best_metric = scan_theta_NH3(T_test, V_total)

    if best_theta is not None:
        eps1_b, eps2_b, eps3_b = best_eps
        print(f"\nScan over theta_NH3 in [2.3, 4.0] at T = {T_test} K, V = {V_total:.2e} L:")
        print(f"  Best theta_NH3       = {best_theta:.3f}")
        print(f"  eps1 (carbamate)     = {eps1_b: .5e}")
        print(f"  eps2 (bicarb)        = {eps2_b: .5e}")
        print(f"  eps3 (urea)          = {eps3_b: .5e}")
        print(f"  Selectivity metric   = eps3/eps2 = {best_metric:.3f}")
    else:
        print("\nScan over theta_NH3: no feasible solution found in [2.3, 4.0].")
