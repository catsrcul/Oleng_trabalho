from analytic_formulation import OlengEquations
from analytic_formulation import ureg, Q_
import numpy as np

# --- Params ---
params = {
    'rd':      Q_(0.316, 'm'),
    'nP':      Q_(4000, 'rpm'),
    'a':       1.2,
    'b':       1.0,
    'c':       0.5,
    'PP':      Q_(100.0, 'kW'),
    'mv':      Q_(1500.0, 'kg'),
    'g':       Q_(9.81, 'm/s**2'),
    'fr':      0.015,                    # plain float, avoids pint friction
    'rho_a':   Q_(1.225, 'kg/m**3'),
    'CD':      0.30,                     # plain float
    'Af':      Q_(2.2, 'm**2'),
    'gamma_m': 1.05,                     # plain float
    'eta_T':   0.92,                     # plain float
    'eta_o':   0.38,
    'HL':      Q_(44e6, 'J/kg'),
}

xi_ax = 3.9
xi_n  = 1.0

# --- Route ---
delta_s    = Q_(500.0, 'm')
total_dist = Q_(100_000.0, 'm')
n_steps    = int(total_dist / delta_s)
v_cruise   = Q_(90.0, 'km/h').to('m/s')

# --- Run ---

model = OlengEquations(params, initial_velocity=Q_(90.0, 'km/h').to('m/s'))

# Constant speed → acceleration = 0 everywhere
n_steps     = 200
delta_s     = Q_(500.0, 'm')
accelerations = np.zeros(n_steps)

df = model.run(accelerations, delta_s, xi_ax=3.9, xi_n=1.0, output_path="results.csv")

print(df[['distance_m', 'velocity_kmh', 'Pi_kW', 'efficiency', 'fuel_cumulative_kg']].tail())