import numpy as np
from analytic_formulation import OlengEquations, Q_
from gcmma_simples import Otimizador, ProblemaOtimizacao

# ... (Keep your existing params dictionary, delta_s, total_dist, etc.) ...
# --- Params ---
# for a toyota yaris
params = {
    'rd':      Q_(0.30, 'm'),
    'nP':      Q_(6000, 'rpm'),
    'a':       1.0,
    'b':       1.0,
    'c':       1.0,
    'PP':      Q_(73.1, 'kW'),
    'mv':      Q_(1005.0, 'kg'),
    'g':       Q_(9.81, 'm/s**2'),
    'fr':      0.015,                    # plain float, avoids pint friction
    'rho_a':   Q_(1.225, 'kg/m**3'),
    'CD':      0.30,                     # plain float
    'Af':      Q_(1.80, 'm**2'),
    'gamma_m': 1.08,                     # plain float
    'eta_T':   0.92,                     # plain float
    'eta_o':   0.38,
    'HL':      Q_(43.4e6, 'J/kg'),
}


# Ensure n_steps is defined (e.g., int(total_dist / delta_s))
delta_s = Q_(500.0, 'm')
total_dist = Q_(10_000.0, 'm')
n_steps = int(total_dist / delta_s)

v_cruise = Q_(90, 'km/h').to('m/s')
v_max = Q_(120, 'km/h').to('m/s')
v_min = Q_(50, 'km/h').to('m/s')
n_min = 1000.0
n_max = 6000.0
a_max = 4.0


class OtimizacaoVeiculo(ProblemaOtimizacao):
    def __init__(self):
        self._n_vars = n_steps
        # We have 3 constraints per step (v_i, n_i, Pi - Pe)
        self._n_restricoes = 3 * n_steps

        # Cache results to prevent running the simulation twice per iteration
        # (once for objective, once for constraints)
        self.last_x = None
        self.last_results = None
        self.last_fuel = None

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def n_restricoes(self) -> int:
        return self._n_restricoes

    def _run_sim(self, x: np.ndarray):
        """Helper to run simulation and cache results."""
        # Only run if x has changed
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return

        try:
            model = OlengEquations(params, initial_velocity=v_cruise)
            results = model.run(x, delta_s, xi_ax=3.9)
            fuel = float(results['fuel_cumulative_kg'].iloc[-1])

            if fuel < 0 or np.isnan(fuel) or np.isinf(fuel):
                raise ValueError("Invalid fuel calculation")

            self.last_results = results
            self.last_fuel = fuel
            self.last_x = x.copy()

        except (ValueError, RuntimeError):
            # Penalty for failed simulation
            self.last_results = None
            self.last_fuel = 1e10 + np.sum(x ** 2)
            self.last_x = x.copy()

    def objetivo(self, x: np.ndarray) -> float:
        self._run_sim(x)
        return self.last_fuel

    def restricoes(self, x: np.ndarray) -> dict:
        self._run_sim(x)
        g_dict = {}

        if self.last_results is None:
            # If sim failed, return highly violated constraints
            for i in range(self.n_restricoes):
                g_dict[f'g{i + 1}'] = 1e5 + np.sum(x ** 2)
            return g_dict

        # Extract variables (skipping initial state at index 0)
        speeds = self.last_results['velocity_ms'].iloc[1:].to_numpy()
        rpms = self.last_results['engine_rpm'].iloc[1:].to_numpy()
        p_req = self.last_results['Pi_kW'].iloc[1:].to_numpy()
        p_avail = self.last_results['Pe_kW'].iloc[1:].to_numpy()

        # GCMMA Convention: g(x) <= 0 is feasible
        idx = 1
        for i in range(self.n_vars):
            # 1. Speed constraint: v_min <= v_i <= v_max
            # Two inequalities: v_min - v_i <= 0 AND v_i - v_max <= 0
            # To simplify into one constraint per step, we can use the max violation:
            g_speed = max(v_min.magnitude - speeds[i], speeds[i] - v_max.magnitude)
            g_dict[f'g{idx}'] = g_speed
            idx += 1

            # 2. RPM constraint: n_min <= n_i <= n_max
            g_rpm = max(n_min - rpms[i], rpms[i] - n_max)
            g_dict[f'g{idx}'] = g_rpm
            idx += 1

            # 3. Power constraint: Pi(v_i) - Pe(n_i) <= 0
            g_dict[f'g{idx}'] = p_req[i] - p_avail[i]
            idx += 1

        return g_dict

    def limites(self) -> tuple:
        xmin = np.full(self.n_vars, -a_max)
        xmax = np.full(self.n_vars, a_max)
        return xmin, xmax

    def inicial(self) -> np.ndarray:
        return np.zeros(self.n_vars)


import matplotlib.pyplot as plt
from analytic_formulation import OlengEquations

# --- Execution ---
problema = OtimizacaoVeiculo()
otimizador = Otimizador(OtimizacaoVeiculo, move=0.1, rho=50, tol=1e-4, max_iter=100)

print("Starting GCMMA optimization...")
x_otimo = otimizador.otimizar()
otimizador.relatorio()

# 1. Plot the GCMMA convergence history
fig_opt = otimizador.graficar()
fig_opt.savefig("gcmma_convergence.png")
print("\n✓ Convergence plot saved to gcmma_convergence.png")

# 2. Run the vehicle model one final time with the optimal solution
final_model = OlengEquations(params, initial_velocity=v_cruise)
final_results = final_model.run(x_otimo, delta_s, xi_ax=3.9)

# 3. Generate the Vehicle Simulation Plots
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
    # Convert velocity to km/h for the plot, keep everything else as is
    y_data = final_results[y_col] * 3.6 if y_col == 'velocity_ms' else final_results[y_col]

    ax.plot(final_results[x_col], y_data, marker='o', markersize=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.savefig("vehicle_results.png")
print("✓ Vehicle simulation plots saved to vehicle_results.png")

final_results.to_csv("results.csv", index=False)
print("✓ Results saved to results.csv")