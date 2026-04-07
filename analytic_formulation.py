import numpy as np
from dataclasses import dataclass
from pint import UnitRegistry

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
    Defined equatins: 10, 11, 12
    Total number of equations: 11
    """

    def calculate_ni(self, v, xi_ax, xi_n):
        """Equation (11) - Rotational speed calculation"""
        v_kmh = v.to('km/h').magnitude
        ni_mag = (9.55 * v_kmh * xi_ax * xi_n) / self.p['rd'].to('m').magnitude
        return Q_(ni_mag, 'rpm')

    def calculate_Pe(self, ni):
        """Equation (10) - Available engine power"""
        ratio = (ni / self.p['nP']).to('dimensionless').magnitude
        terms = (self.p['a'] * ratio +
                 self.p['b'] * ratio ** 2 -
                 self.p['c'] * ratio ** 3)
        return self.p['PP'].to('kW') * terms

    def calculate_Pi(self, v, a):
        """Equation (12) - Required instantaneous power"""
        v_ms = v.to('m/s')
        a_ms2 = a.to('m/s**2')

        # Resistance forces * velocity
        # Here the velocities aren't divided by 3.6 because "pint" already takes care of unit conversion.
        rolling_res = self.p['mv'] * self.p['g'] * self.p['fr'] * v_ms
        aero_res = 0.5 * self.p['rho_a'] * self.p['CD'] * self.p['Af'] * (v_ms ** 3)

        # The inertial part in the paper is wrong, we need to convert the [N] we end up with to th kW we want by
        # multiplying by the velocity.
        inertial_res = self.p['mv'] * self.p['gamma_m'] * a_ms2 * v_ms

        total_power = (1 / self.p['eta_T']) * (rolling_res + aero_res + inertial_res)
        return total_power.to('kW')

    def calculate_total_efficiency(self, v, a, xi_ax, xi_n):
        """Equations (12-14) - System efficiency lookup"""
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

        # Eq 12 - Correction factor for load (mu_P)
        mu_P = 0.5968 - 0.1666 * r_P + 2.4968 * (r_P ** 2) - 2.1128 * (r_P ** 3)

        # Eq 13 - Correction factor for speed (mu_n)
        mu_n = 0.7107 + 0.9963 * r_n - 1.0582 * (r_n ** 2) + 0.3124 * (r_n ** 3)

        # Eq 14 - Final combined efficiency
        eta = self.p['eta_o'] * mu_P * mu_n
        return max(float(eta), 0.0), Pi
