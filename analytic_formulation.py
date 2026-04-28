import csv
import logging
from dataclasses import dataclass
from pint import UnitRegistry
import numpy as np

try:
    import pandas as pd
except ImportError:
    class _FallbackDataFrame:
        def __init__(self, rows):
            self._rows = list(rows)

        def to_csv(self, output_path, index=False):
            fieldnames = list(self._rows[0].keys()) if self._rows else []
            with open(output_path, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._rows)

    class _FallbackPandas:
        @staticmethod
        def DataFrame(rows):
            return _FallbackDataFrame(rows)

    pd = _FallbackPandas()

logger = logging.getLogger(__name__)

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
    power: ureg.Quantity  # Required power Pi in this segment (kW), evaluated at v_avg
    fuel_consumed: ureg.Quantity  # Fuel consumed in this segment (kg or L)
    # Intermediates evaluated at v_avg (average segment velocity) for consistent logging
    v_avg: ureg.Quantity = None  # Average velocity used for all power/efficiency calculations
    engine_rpm: ureg.Quantity = None  # Engine speed at v_avg
    Pe: ureg.Quantity = None  # Available engine power at v_avg
    r_P: float = 0.0  # Power ratio Pi/Pe at v_avg
    r_n: float = 0.0  # Speed ratio ni/nP at v_avg
    mu_P: float = 0.0  # Load correction factor at v_avg
    mu_n: float = 0.0  # Speed correction factor at v_avg


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

    @staticmethod
    def get_xi_n(velocity_ms):
        v_kmh = velocity_ms.to('km/h').magnitude

        if v_kmh < 20.0:
            return 3.5  # 1st gear
        elif v_kmh < 40.0:
            return 2.1  # 2nd gear
        elif v_kmh < 60.0:
            return 1.4  # 3rd gear
        elif v_kmh < 80.0:
            return 1.0  # 4th gear
        else:
            return 0.8  # 5th gear


    def step_distance(self, delta_s, v_target, xi_ax):
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

        # getting xi_n -- gear ratio
        xi_n = self.get_xi_n(v_avg_safe)


        # Constant acceleration derived from distance and velocities: a = (v_f^2 - v_i^2) / (2 * ds)
        # Note: We use .to('m/s') to ensure units cancel out correctly for the magnitude
        accel = (v_target ** 2 - v_start ** 2) / (2 * delta_s)

        # 2. Power and Efficiency (Equations 9-14)
        # We evaluate the vehicle state at the average velocity of the segment
        eta, Pi, ni, Pe, r_P, r_n, mu_P, mu_n = self.calculate_total_efficiency(v_avg, accel, xi_ax, xi_n)

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
            fuel_consumed=instant_fuel,
            v_avg=v_avg,
            engine_rpm=ni,
            Pe=Pe,
            r_P=r_P,
            r_n=r_n,
            mu_P=mu_P,
            mu_n=mu_n,
        )

        self.history.append(record)
        return record

    # --- Core Mathematical Formulations ---

    """
    Implemented equations: 6, 7, 9 (efficiency), 10, 11, 12
    """

    def calculate_total_efficiency(self, v, a, xi_ax, xi_n):
        """Equations (6-9) - instantaneous System efficiency lookup.

        Returns (eta, Pi, ni, Pe, r_P, r_n, mu_P, mu_n) all evaluated at velocity v.
        """
        zero_kW = Q_(0.0, 'kW')
        zero_rpm = Q_(0.0, 'rpm')
        if v < Q_(0.1, 'km/h'):
            return 0.0, zero_kW, zero_rpm, zero_kW, 0.0, 0.0, 0.0, 0.0

        ni = self.calculate_ni(v, xi_ax, xi_n)
        Pe = self.calculate_Pe(ni)
        Pi = self.calculate_Pi(v, a)
        r_n = float((ni / self.p['nP']).to('dimensionless').magnitude)

        if Pe <= Q_(0, 'kW') or Pi <= Q_(0, 'kW'):
            return 0.0, Pi, ni, Pe, 0.0, r_n, 0.0, 0.0

        # Non-dimensional ratios for the empirical polynomial
        r_P = float((Pi / Pe).to('dimensionless').magnitude)

        # Eq 7 - Correction factor for load (mu_P)
        mu_P = 0.5968 - 0.1666 * r_P + 2.4968 * (r_P ** 2) - 2.1128 * (r_P ** 3)

        # Eq 9 - Correction factor for speed (mu_n)
        mu_n = 0.7107 + 0.9963 * r_n - 1.0582 * (r_n ** 2) + 0.3124 * (r_n ** 3)

        # Eq 6 - Final combined efficiency
        # where eta_o is the peak efficiency of the engine
        eta = self.p['eta_o'] * mu_P * mu_n
        return max(float(eta), 0.0), Pi, ni, Pe, r_P, r_n, float(mu_P), float(mu_n)

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



    def run(self, accelerations, delta_s, xi_ax, output_path=None):
        """
        Run full simulation from acceleration array.

        accelerations: list/array of floats (m/s^2) or Pint Quantities, one per distance step
        delta_s: Quantity (m), fixed step size
        xi_ax, xi_n: floats, gear/axle ratios
        output_path: str or None; if given, results are saved as CSV

        Returns: pandas DataFrame with all computed quantities.
            Row 0 is the initial state (distance=0, values from __init__).
            Rows 1..N correspond to each acceleration step.
        """
        # Reset history to initial state only
        self.history = [self.history[0]]

        ds = delta_s.to('m').magnitude

        for a_val in accelerations:
            # Accept both plain floats and Pint Quantities
            if isinstance(a_val, ureg.Quantity):
                a_val = a_val.to('m/s**2').magnitude

            prev = self.history[-1]
            v_start = prev.velocity.to('m/s').magnitude

            # Derive v_target from kinematics: v_f = sqrt(v_i^2 + 2*a*ds)
            v_f_sq = v_start ** 2 + 2 * a_val * ds

            # Reject physically impossible segments (stopped at both ends)
            if ds > 0.0 and np.isclose(v_start, 0.0) and np.isclose(max(v_f_sq, 0.0), 0.0):
                raise ValueError(
                    "Invalid segment in run(): positive distance step cannot be "
                    "covered with both start and end velocity equal to zero. "
                    "Provide a non-zero initial velocity or acceleration."
                )

            v_f_sq = max(v_f_sq, 0.0)  # guard negative under sqrt
            v_target = Q_(np.sqrt(v_f_sq), 'm/s')

            self.step_distance(delta_s, v_target, xi_ax)

        # --- Build DataFrame ---
        rows = []
        for r in self.history:
            # Use stored intermediates — all evaluated at v_avg — for consistent logging
            v_avg_ms = r.v_avg.to('m/s').magnitude if r.v_avg is not None else 0.0
            ni_rpm = r.engine_rpm.to('rpm').magnitude if r.engine_rpm is not None else 0.0
            Pe_kW = r.Pe.to('kW').magnitude if r.Pe is not None else 0.0
            Pi_kW = r.power.to('kW').magnitude
            energy_j = (r.power * r.delta_t).to('joule').magnitude if r.power.magnitude > 0 else 0.0

            rows.append({
                'distance_m': r.distance.to('m').magnitude,
                'time_s': r.time_index.to('s').magnitude,
                'delta_t_s': r.delta_t.to('s').magnitude,
                'velocity_ms': r.velocity.to('m/s').magnitude,
                'velocity_kmh': r.velocity.to('km/h').magnitude,
                'v_avg_ms': v_avg_ms,
                'acceleration_ms2': r.acceleration.to('m/s**2').magnitude,
                'engine_rpm': ni_rpm,
                'Pe_kW': Pe_kW,
                'Pi_kW': Pi_kW,
                'r_P': r.r_P,
                'r_n': r.r_n,
                'mu_P': r.mu_P,
                'mu_n': r.mu_n,
                'efficiency': r.efficiency,
                'energy_J': energy_j,
                'fuel_kg': r.fuel_consumed.to('kg').magnitude,
            })


        data_frame = pd.DataFrame(rows)

        data_frame['fuel_cumulative_kg'] = data_frame['fuel_kg'].cumsum()
        # useful for sanity check on results
        if output_path is not None:
            data_frame.to_csv(output_path, index=False)
            logger.debug("Saved %d rows → %s", len(data_frame), output_path)

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
        model.step_distance(delta_s, v_cruise, xi_ax)

    # --- Results ---
    total_fuel = sum(r.fuel_consumed.to('kg').magnitude for r in model.history)
    total_time = model.history[-1].time_index.to('s').magnitude / 3600  # hours

    print(f"Total fuel:  {total_fuel:.3f} kg")
    print(f"Total time:  {total_time:.2f} h")
    print(f"Consumption: {total_fuel / (total_dist.to('km').magnitude) * 100:.2f} kg/100km")
