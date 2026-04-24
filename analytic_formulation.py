from dataclasses import dataclass
from pint import UnitRegistry
import pandas as pd
import numpy as np

# Initialize unit registry for physical consistency
ureg = UnitRegistry()
Q_ = ureg.Quantity


@dataclass
class StateRecord:
    """Stores the state of the vehicle at a specific distance marker."""
    distance: ureg.Quantity  # Fixed step markers (e.g., meters)
    velocity: ureg.Quantity  # Velocity at this marker (m/s)
    delta_t: ureg.Quantity  # Time taken to reach this marker from previous (s)
    time_index: ureg.Quantity  # Cumulative time (s)
    acceleration: ureg.Quantity  # Constant acceleration in this segment (m/s^2)
    efficiency: float  # System efficiency in this segment [0-1]
    power: ureg.Quantity  # Average power in this segment (kW)
    fuel_consumed: ureg.Quantity  # Fuel consumed in this segment (kg or L)


class OlengEquations:
    def __init__(self, params, initial_velocity=0.0):
        """
        Initializes the vehicle model with parameters and starting state.

        params: dictionary containing Quantities for:
            rd (m), nP (rpm), a, b, c (dim), PP (kW), mv (kg), g (m/s^2),
            fr (dim), rho_a (kg/m^3), CD (dim), Af (m^2), gamma_m (dim),
            eta_T (dim), eta_o (dim), HL (J/kg or J/L)
        """
        self.p = params
        self.history = []

        # Ensure initial velocity is a Quantity
        v0 = initial_velocity if isinstance(initial_velocity, ureg.Quantity) else Q_(initial_velocity, 'm/s')

        # Initial State (Marker 0)
        self.history.append(StateRecord(
            distance=Q_(0.0, 'm'),
            velocity=v0,
            delta_t=Q_(0.0, 's'),
            time_index=Q_(0.0, 's'),
            acceleration=Q_(0.0, 'm/s**2'),
            efficiency=0.0,
            power=Q_(0.0, 'kW'),
            fuel_consumed=Q_(0.0, 'kg')
        ))

    def step_distance(self, delta_s, v_target, xi_ax, xi_n):
        """
        Calculates the transition to the next distance marker.

        delta_s: Fixed distance step (Quantity, e.g., 100m)
        v_target: Target velocity at the end of the distance step (Quantity, e.g., 20m/s)
        xi_ax, xi_n: Transmission ratios (floats)
        """
        prev = self.history[-1]
        v_start = prev.velocity

        # 1. Kinematics for a fixed distance step
        # v_avg is used for power calculations and time estimation
        v_avg = (v_start + v_target) / 2

        # Guard against zero velocity to avoid infinite time
        v_avg_safe = max(v_avg, Q_(1e-3, 'm/s'))
        dt = delta_s / v_avg_safe

        # Constant acceleration derived from distance and velocities: a = (v_f^2 - v_i^2) / (2 * ds)
        # Note: We use .to('m/s') to ensure units cancel out correctly for the magnitude
        accel = (v_target ** 2 - v_start ** 2) / (2 * delta_s)

        # 2. Power and Efficiency (Equations 9-14)
        # We evaluate the vehicle state at the average velocity of the segment
        eta, Pi = self.calculate_total_efficiency(v_avg, accel, xi_ax, xi_n)

        # 3. Energy and Consumption
        # Energy = Power * Time
        energy = (Pi * dt).to('joule')

        if Pi <= Q_(0, 'kW') or eta <= 0:
            instant_fuel = Q_(0.0, 'kg')
        else:
            # Fuel = Energy / (Lower Heating Value * Efficiency)
            instant_fuel = energy / (self.p['HL'] * eta)

        record = StateRecord(
            distance=prev.distance + delta_s,
            velocity=v_target,
            delta_t=dt,
            time_index=prev.time_index + dt,
            acceleration=accel,
            efficiency=eta,
            power=Pi.to('kW'),
            fuel_consumed=instant_fuel
        )

        self.history.append(record)
        return record

    # --- Core Mathematical Formulations ---

    """
    Implemented equations: 6, 7, 9 (efficiency), 10, 11, 12
    """

    def calculate_total_efficiency(self, v, a, xi_ax, xi_n):
        """Equations (6-9) - instantaneous System efficiency lookup"""
        if v < Q_(0.1, 'km/h'):
            return 0.0, Q_(0.0, 'kW')

        ni = self.calculate_ni(v, xi_ax, xi_n)
        Pe = self.calculate_Pe(ni)
        Pi = self.calculate_Pi(v, a)

        if Pe <= Q_(0, 'kW') or Pi <= Q_(0, 'kW'):
            return 0.0, Pi

        # Non-dimensional ratios for the empirical polynomial
        r_P = (Pi / Pe).to('dimensionless').magnitude
        r_n = (ni / self.p['nP']).to('dimensionless').magnitude

        # Eq 7 - Correction factor for load (mu_P)
        mu_P = 0.5968 - 0.1666 * r_P + 2.4968 * (r_P ** 2) - 2.1128 * (r_P ** 3)

        # Eq 9 - Correction factor for speed (mu_n)
        mu_n = 0.7107 + 0.9963 * r_n - 1.0582 * (r_n ** 2) + 0.3124 * (r_n ** 3)

        # Eq 6 - Final combined efficiency
        # where eta_o is the peak efficiency of the engine
        eta = self.p['eta_o'] * mu_P * mu_n
        return max(float(eta), 0.0), Pi

    def calculate_Pe(self, ni):
        """Equation (10) - Available engine power"""
        ratio = (ni / self.p['nP']).to('dimensionless').magnitude
        terms = (self.p['a'] * ratio +
                 self.p['b'] * ratio ** 2 -
                 self.p['c'] * ratio ** 3)
        return self.p['PP'].to('kW') * terms

    def calculate_ni(self, v, xi_ax, xi_n):
        """Equation (11) - Rotational speed calculation"""
        v_kmh = v.to('km/h').magnitude
        ni_mag = (9.55 * v_kmh * xi_ax * xi_n) / self.p['rd'].to('m').magnitude
        return Q_(ni_mag, 'rpm')

    def calculate_Pi(self, v, a):
        """Equation (12) - Required instantaneous power"""
        v_ms = v.to('m/s')
        a_ms2 = a.to('m/s**2')

        # Resistance forces * velocity
        # Here the velocities aren't divided by 3.6 because "pint" already takes care of unit conversion.
        rolling_res = self.p['mv'] * self.p['g'] * float(self.p['fr']) * v_ms
        aero_res = 0.5 * self.p['rho_a'] * self.p['CD'] * self.p['Af'] * (v_ms ** 3)

        # The inertial term in the paper yields force [N]; multiply by velocity to convert it to power.
        inertial_res = self.p['mv'] * self.p['gamma_m'] * a_ms2 * v_ms

        total_power = (1 / self.p['eta_T']) * (rolling_res + aero_res + inertial_res)
        return total_power.to('kW')



    def run(self, accelerations, delta_s, xi_ax, xi_n, output_path=None):
        """
        Run full simulation from acceleration array.

        accelerations: list/array of floats (m/s^2), one per distance step
        delta_s: Quantity (m), fixed step size
        xi_ax, xi_n: floats, gear/axle ratios
        output_path: str, where to save CSV

        Returns: pandas DataFrame with all computed quantities
        """
        # Reset history to initial state only
        self.history = [self.history[0]]

        for a_val in accelerations:
            prev = self.history[-1]
            v_start = prev.velocity.to('m/s').magnitude

            # Derive v_target from kinematics: v_f = sqrt(v_i^2 + 2*a*ds)
            ds = delta_s.to('m').magnitude
            v_f_sq = v_start ** 2 + 2 * a_val * ds
            v_f_sq = max(v_f_sq, 0.0)  # guard negative under sqrt
            v_target = Q_(np.sqrt(v_f_sq), 'm/s')

            self.step_distance(delta_s, v_target, xi_ax, xi_n)

        # --- Build DataFrame ---
        rows = []
        for r in self.history:
            v = r.velocity.to('m/s').magnitude
            a = r.acceleration.to('m/s**2').magnitude
            xi_ax_local = xi_ax
            xi_n_local = xi_n

            # Recompute intermediate quantities for logging
            ni = self.calculate_ni(r.velocity, xi_ax_local, xi_n_local)
            Pe = self.calculate_Pe(ni)
            Pi = self.calculate_Pi(r.velocity, r.acceleration)
            r_P = float((Pi / Pe).to('dimensionless').magnitude) if Pe.magnitude > 0 else 0.0
            r_n = float((ni / self.p['nP']).to('dimensionless').magnitude)
            mu_P = 0.5968 - 0.1666 * r_P + 2.4968 * r_P ** 2 - 2.1128 * r_P ** 3
            mu_n = 0.7107 + 0.9963 * r_n - 1.0582 * r_n ** 2 + 0.3124 * r_n ** 3

            energy_j = (Pi * r.delta_t).to('joule').magnitude if Pi.magnitude > 0 else 0.0

            rows.append({
                'distance_m': r.distance.to('m').magnitude,
                'time_s': r.time_index.to('s').magnitude,
                'delta_t_s': r.delta_t.to('s').magnitude,
                'velocity_ms': v,
                'velocity_kmh': r.velocity.to('km/h').magnitude,
                'acceleration_ms2': a,
                'engine_rpm': ni.to('rpm').magnitude,
                'Pe_kW': Pe.to('kW').magnitude,
                'Pi_kW': Pi.to('kW').magnitude,
                'r_P': r_P,
                'r_n': r_n,
                'mu_P': mu_P,
                'mu_n': mu_n,
                'efficiency': r.efficiency,
                'energy_J': energy_j,
                'fuel_kg': r.fuel_consumed.to('kg').magnitude,
            })


        data_frame = pd.DataFrame(rows)

        data_frame['fuel_cumulative_kg'] = data_frame['fuel_kg'].cumsum()
        # useful for sanity check on results
        if output_path is not None:
            data_frame.to_csv(output_path, index=False)
            print(f"Saved {len(data_frame)} rows → {output_path}")

        return data_frame

if __name__ == "__main__":
    # Reuse the module-level unit registry/quantity constructor.

    # --- Vehicle Parameters (typical passenger car) ---
    params = {
        'rd':      Q_(0.316, 'm'),           # Wheel radius
        'nP':      Q_(4000, 'rpm'),           # Rated engine speed
        'a':       1.2,                       # Engine polynomial coeff a
        'b':       1.0,                       # Engine polynomial coeff b
        'c':       0.5,                       # Engine polynomial coeff c
        'PP':      Q_(100.0, 'kW'),           # Peak engine power
        'mv':      Q_(1500.0, 'kg'),          # Vehicle mass
        'g':       Q_(9.81, 'm/s**2'),
        'fr':      Q_(0.015, 'dimensionless'),# Rolling resistance
        'rho_a':   Q_(1.225, 'kg/m**3'),     # Air density
        'CD':      Q_(0.30, 'dimensionless'), # Drag coefficient
        'Af':      Q_(2.2, 'm**2'),           # Frontal area
        'gamma_m': Q_(1.05, 'dimensionless'),# Rotational inertia factor
        'eta_T':   Q_(0.92, 'dimensionless'),# Drivetrain efficiency
        'eta_o':   Q_(0.38, 'dimensionless'),# Peak engine efficiency
        'HL':      Q_(44e6, 'J/kg'),         # Lower heating value (gasoline)
    }

    # Gear/axle ratios (fix these to a single gear for now)
    xi_ax = 3.9   # Axle ratio
    xi_n  = 1.0   # Gear ratio (top gear)

    # --- Route: 100 km at constant 90 km/h ---
    delta_s   = Q_(500.0, 'm')          # Distance step
    total_dist = Q_(100_000.0, 'm')     # 100 km
    n_steps   = int(total_dist / delta_s)
    v_cruise  = Q_(90.0, 'km/h').to('m/s')

    model = OlengEquations(params, initial_velocity=v_cruise)

    for _ in range(n_steps):
        model.step_distance(delta_s, v_cruise, xi_ax, xi_n)

    # --- Results ---
    total_fuel = sum(r.fuel_consumed.to('kg').magnitude for r in model.history)
    total_time = model.history[-1].time_index.to('s').magnitude / 3600  # hours

    print(f"Total fuel:  {total_fuel:.3f} kg")
    print(f"Total time:  {total_time:.2f} h")
    print(f"Consumption: {total_fuel / (total_dist.to('km').magnitude) * 100:.2f} kg/100km")
