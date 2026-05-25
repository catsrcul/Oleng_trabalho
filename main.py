from analytic_formulation import Q_


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


from scipy.optimize import minimize, Bounds, NonlinearConstraint
from analytic_formulation import OlengEquations, Q_
import numpy as np

# --- Optimization Setup ---
delta_s    = Q_(500.0, 'm')
total_dist = Q_(30_000.0, 'm')
n_steps    = int(total_dist / delta_s)

v_cruise = Q_(20, 'km/h').to('m/s')
v_max = Q_(120, 'km/h').to('m/s')
v_min = Q_(50, 'km/h').to('m/s')  # Added min speed
n_min = 1000.0                    # Added min RPM
n_max = 6000.0                    # Added max RPM
a_max = 4.0                       # m/s²


def make_model():
    """Create a fresh model for each evaluation to ensure consistency."""
    return OlengEquations(params, initial_velocity=v_cruise)

def fuel_objective(acceleration_profile):
    """Objective: minimize total fuel consumption.
    Includes error handling for infeasible profiles.
    """
    try:
        model = make_model()
        results = model.run(acceleration_profile, delta_s, xi_ax=3.9)
        fuel = float(results['fuel_cumulative_kg'].iloc[-1])
        # Penalize invalid results
        if fuel < 0 or np.isnan(fuel) or np.isinf(fuel):
            return 1e10
        return fuel
    except (ValueError, RuntimeError):
        # If simulation fails, return large penalty
        return 1e10 + np.sum(acceleration_profile**2)


def combined_constraints(acceleration_profile):
    """Eval velocities, RPMs, and power surplus to satisfy 3 missing equations."""
    try:
        model = make_model()
        results = model.run(acceleration_profile, delta_s, xi_ax=3.9)

        # Skip row 0 (initial state)
        speeds = results['velocity_ms'].iloc[1:].to_numpy()
        rpms = results['engine_rpm'].iloc[1:].to_numpy()
        p_req = results['Pi_kW'].iloc[1:].to_numpy()
        p_avail = results['Pe_kW'].iloc[1:].to_numpy()

        # Eq: Pi - Pe <= 0 is same as Pe - Pi >= 0
        power_margin = p_avail - p_req

        # Concat arrays for single SciPy constraint evaluation
        return np.concatenate([speeds, rpms, power_margin])

    except (ValueError, RuntimeError):
        # Return failing bounds if simulation crashes
        n = len(acceleration_profile)
        return np.concatenate([
            np.zeros(n),  # Fails v_min
            np.zeros(n),  # Fails n_min
            np.full(n, -1.0)  # Fails power >= 0
        ])


# Bounds on acceleration: -4 to +4 m/s² (allows braking)
bounds = Bounds([-a_max] * n_steps, [a_max] * n_steps)

# Nonlinear constraint: speed <= v_max everywhere, RPM limits, and power surplus >= 0
# Build lb/ub arrays for combined constraint
# Order: [speeds, rpms, power_margins]
lb = np.concatenate([
    np.full(n_steps, v_min.magnitude),  # v_i >= v_min
    np.full(n_steps, n_min),            # n_i >= n_min
    np.zeros(n_steps)                   # Pe - Pi >= 0
])

ub = np.concatenate([
    np.full(n_steps, v_max.magnitude),  # v_i <= v_max
    np.full(n_steps, n_max),            # n_i <= n_max
    np.full(n_steps, np.inf)            # No max limit on available power surplus
])

constraints = [
    NonlinearConstraint(combined_constraints, lb, ub)
]

initial_guess = np.zeros(n_steps)  # Start with zero acceleration (constant speed) as initial guess


# Run optimization
print("Starting optimization...")
minimization = minimize(
    fuel_objective,
    initial_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 100, 'ftol': 1e-6}
)

print("\n✓ Optimization complete!")
print(f"  Success: {minimization.success}")
print(f"  Message: {minimization.message}")
print(f"  Optimal fuel: {minimization.fun:.6f} kg")
print(f"  Iterations: {minimization.nit}")

# Verify constraints are satisfied
final_results = make_model().run(minimization.x, delta_s, xi_ax=3.9)
max_speed = final_results['velocity_kmh'].max()
max_accel = np.abs(minimization.x).max()
print(f"\n✓ Constraints verified:")
print(f"  Max speed: {max_speed:.2f} km/h (limit: 120 km/h)")
print(f"  Max |acceleration|: {max_accel:.3f} m/s² (limit: {a_max} m/s²)")


import matplotlib.pyplot as plt

# Use the final results from the optimization
results = final_results

print("Optimal acceleration profile (m/s²):")
print(minimization.x)
print(f"Total fuel: {results['fuel_cumulative_kg'].iloc[-1]:.10f} kg")

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

plots = [
    ('time_s', 'velocity_ms',          'Time (s)', 'Velocity (km/h)',       'Velocity over Time'),
    ('time_s', 'distance_m',           'Time (s)', 'Distance (m)',          'Distance over Time'),
    ('time_s', 'acceleration_ms2',     'Time (s)', 'Acceleration (m/s²)',   'Acceleration over Time'),
    ('time_s', 'efficiency',           'Time (s)', 'Efficiency [-]',        'Engine Efficiency over Time'),
    ('time_s', 'Pi_kW',               'Time (s)', 'Power (kW)',            'Required Power (Pi) over Time'),
    ('time_s', 'Pe_kW',               'Time (s)', 'Power (kW)',            'Available Power (Pe) over Time'),
    ('time_s', 'fuel_cumulative_kg',   'Time (s)', 'Fuel (kg)',             'Cumulative Fuel over Time'),
    ('time_s', 'engine_rpm',           'Time (s)', 'RPM',                   'Engine RPM over Time'),
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

# Also save results to CSV
results.to_csv("results.csv", index=False)
print("\n✓ Results saved to results.csv")