import numpy as np
import matplotlib.pyplot as plt
from analytic_formulation import OlengEquations, Q_
from gcmma_simples import ProblemaOtimizacao, Otimizador

# --- Params ---
params = {
    'rd': Q_(0.30, 'm'),
    'nP': Q_(6000, 'rpm'),
    'a': 1.0,
    'b': 1.0,
    'c': 1.0,
    'PP': Q_(73.1, 'kW'),
    'mv': Q_(1005.0, 'kg'),
    'g': Q_(9.81, 'm/s**2'),
    'fr': 0.015,
    'rho_a': Q_(1.225, 'kg/m**3'),
    'CD': 0.30,
    'Af': Q_(1.80, 'm**2'),
    'gamma_m': 1.08,
    'eta_T': 0.92,
    'eta_o': 0.38,
    'HL': Q_(43.4e6, 'J/kg'),
}

# --- Setup ---
delta_s = Q_(500.0, 'm')
total_dist = Q_(10_000.0, 'm')
n_steps = int(total_dist / delta_s)

v_cruise = Q_(50, 'km/h').to('m/s')
v_max = Q_(120, 'km/h').to('m/s')
v_min = Q_(50, 'km/h').to('m/s')
n_min = 1000.0
n_max = 6000.0
a_max = 4.0

# --- Float Constants for Fast Math ---
rd_f = params['rd'].to('m').magnitude
nP_f = params['nP'].to('rpm').magnitude
PP_f = params['PP'].to('kW').magnitude
mv_f = params['mv'].to('kg').magnitude
g_f = params['g'].to('m/s**2').magnitude
rho_f = params['rho_a'].to('kg/m**3').magnitude
Af_f = params['Af'].to('m**2').magnitude
HL_f = params['HL'].to('J/kg').magnitude
ds_f = delta_s.to('m').magnitude
v0_f = v_cruise.to('m/s').magnitude
v_min_f = v_min.magnitude
v_max_f = v_max.magnitude


# --- Fast Simulator (Bypass Pint) ---
def fast_simulate(x):
    """Run physics pure floats. Milliseconds per evaluation."""
    n = len(x)
    speeds = np.zeros(n)
    rpms = np.zeros(n)
    p_margins = np.zeros(n)
    total_fuel = 0.0

    v_curr = v0_f
    for i in range(n):
        a_val = x[i]
        v_next_sq = v_curr ** 2 + 2 * a_val * ds_f
        if v_next_sq < 0:
            return 1e10, np.zeros(n), np.zeros(n), np.full(n, -100.0)

        v_next = np.sqrt(v_next_sq)
        v_avg = (v_curr + v_next) / 2.0
        v_avg_safe = max(v_avg, 1e-3)
        dt = ds_f / v_avg_safe

        # Smooth gears
        v_kmh = v_avg_safe * 3.6
        k = 0.5
        s1 = 0.5 * (1 + np.tanh(k * (v_kmh - 20.0)))
        s2 = 0.5 * (1 + np.tanh(k * (v_kmh - 40.0)))
        s3 = 0.5 * (1 + np.tanh(k * (v_kmh - 60.0)))
        s4 = 0.5 * (1 + np.tanh(k * (v_kmh - 80.0)))
        xi_n = 3.5 + (2.1 - 3.5) * s1 + (1.4 - 2.1) * s2 + (1.0 - 1.4) * s3 + (0.8 - 1.0) * s4

        ni = (9.55 * v_kmh * 3.9 * xi_n) / rd_f
        r_n = ni / nP_f
        Pe = PP_f * (1.0 * r_n + 1.0 * r_n ** 2 - 1.0 * r_n ** 3)

        rolling = mv_f * g_f * 0.015 * v_avg_safe
        aero = 0.5 * rho_f * 0.30 * Af_f * (v_avg_safe ** 3)
        inertial = mv_f * 1.08 * a_val * v_avg_safe
        Pi = (1.0 / 0.92) * (rolling + aero + inertial) / 1000.0

        p_margin = Pe - Pi

        if Pe <= 0 or Pi <= 0:
            fuel_kg = 0.0
        else:
            r_P = Pi / Pe
            mu_P = 0.5968 - 0.1666 * r_P + 2.4968 * (r_P ** 2) - 2.1128 * (r_P ** 3)
            mu_n = 0.7107 + 0.9963 * r_n - 1.0582 * (r_n ** 2) + 0.3124 * (r_n ** 3)
            eta = max(0.38 * mu_P * mu_n, 0.0)

            if eta <= 0:
                fuel_kg = 0.0
            else:
                energy_j = Pi * 1000.0 * dt
                fuel_kg = energy_j / (HL_f * eta)

        speeds[i] = v_next
        rpms[i] = ni
        p_margins[i] = p_margin
        total_fuel += fuel_kg
        v_curr = v_next

    return total_fuel, speeds, rpms, p_margins


# --- GCMMA Problem ---
class ProblemaCombustivel(ProblemaOtimizacao):
    def __init__(self, config_path=None):
        self.num_steps = n_steps
        self._cache_x = None
        self._cache_res = None

    def _run_cached(self, x: np.ndarray):
        """Cache prevents double evaluation per iteration."""
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache_res
        res = fast_simulate(x)
        self._cache_x = x.copy()
        self._cache_res = res
        return res

    def objetivo(self, x: np.ndarray) -> float:
        fuel, _, _, _ = self._run_cached(x)
        return fuel

    def restricoes(self, x: np.ndarray) -> dict:
        _, speeds, rpms, p_margins = self._run_cached(x)
        g_dict = {}
        idx = 1

        for i in range(self.num_steps):
            g_dict[f'g{idx}'] = v_min_f - speeds[i]
            idx += 1
            g_dict[f'g{idx}'] = speeds[i] - v_max_f
            idx += 1
            g_dict[f'g{idx}'] = n_min - rpms[i]
            idx += 1
            g_dict[f'g{idx}'] = rpms[i] - n_max
            idx += 1
            g_dict[f'g{idx}'] = -p_margins[i]
            idx += 1

        return g_dict

    def limites(self) -> tuple:
        xmin = np.full(self.num_steps, -a_max)
        xmax = np.full(self.num_steps, a_max)
        return xmin, xmax

    def inicial(self) -> np.ndarray:
        return np.zeros(self.num_steps)

    @property
    def n_vars(self) -> int:
        return self.num_steps

    @property
    def n_restricoes(self) -> int:
        return self.num_steps * 5


# --- Smooth Equations Override for Final Plotting ---
class OlengEquationsSmooth(OlengEquations):
    @staticmethod
    def get_xi_n(velocity_ms):
        v_kmh = velocity_ms.to('km/h').magnitude
        import numpy as np
        k = 0.5
        s1 = 0.5 * (1 + np.tanh(k * (v_kmh - 20.0)))
        s2 = 0.5 * (1 + np.tanh(k * (v_kmh - 40.0)))
        s3 = 0.5 * (1 + np.tanh(k * (v_kmh - 60.0)))
        s4 = 0.5 * (1 + np.tanh(k * (v_kmh - 80.0)))
        g1, g2, g3, g4, g5 = 3.5, 2.1, 1.4, 1.0, 0.8
        return float(g1 + (g2 - g1) * s1 + (g3 - g2) * s2 + (g4 - g3) * s3 + (g5 - g4) * s4)


# --- Execute ---
if __name__ == "__main__":
    print("Start GCMMA...")
    otimizador = Otimizador(
        ProblemaCombustivel,
        move=0.2,
        max_iter=50,
        verbose=True
    )

    x_otimo = otimizador.otimizar()
    print("\n✓ Otimização GCMMA concluída!")

    # Plot real model
    model = OlengEquationsSmooth(params, initial_velocity=v_cruise)
    results = model.run(x_otimo, delta_s, xi_ax=3.9)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    plots = [
        ('time_s', 'velocity_ms', 'Time (s)', 'Velocity (km/h)', 'Velocity over Time'),
        ('time_s', 'distance_m', 'Time (s)', 'Distance (m)', 'Distance over Time'),
        ('time_s', 'acceleration_ms2', 'Time (s)', 'Acceleration (m/s²)', 'Acceleration over Time'),
        ('time_s', 'efficiency', 'Time (s)', 'Efficiency [-]', 'Engine Efficiency over Time'),
        ('time_s', 'Pi_kW', 'Time (s)', 'Power (kW)', 'Required Power (Pi) over Time'),
        ('time_s', 'Pe_kW', 'Time (s)', 'Power (kW)', 'Available Power (Pe) over Time'),
        ('time_s', 'fuel_cumulative_kg', 'Time (s)', 'Fuel (kg)', 'Cumulative Fuel over Time'),
        ('time_s', 'engine_rpm', 'Time (s)', 'RPM', 'Engine RPM over Time'),
    ]

    for ax, (x_col, y_col, x_label, y_label, title) in zip(axes, plots):
        y_data = results[y_col] * 3.6 if y_col == 'velocity_ms' else results[y_col]
        ax.plot(results[x_col], y_data, marker='o', markersize=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    otimizador.graficar()
    plt.show()