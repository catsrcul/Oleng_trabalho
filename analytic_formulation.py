import numpy as np
from state_record import StateRecord

from dataclasses import asdict

## first draft with a linear formulation

class OlengEquations:
    def __init__(self, params, initial_velocity=0.0):
        """
        params: dictionary with constants (m_v, g, f_r, rho_a, Cd, Af, etc.)
        """
        self.p = params
        self.history = []

        # Initial State at t=0
        self.history.append(StateRecord(
            time_index=Q_(0.0, 's'),
            delta_t=Q_(0.0, 's'),
            acceleration=Q_(0.0, 'm/s**2'),
            velocity=v0,
            distance=Q_(0.0, 'm'),
            efficiency=0.0,
            power=Q_(0.0, 'kW'),
            fuel_consumed=Q_(0.0, 'kg')
        ))



    def step(self, delta_t, a_i, xi_ax, xi_n):
        """
        Calculates and stores the current moment's data based on the equations.
        """
        prev = self.history[-1]

        # 1. Update Kinematics
        v_i = prev.velocity + (a_i * delta_t)
        # Ensure velocity doesn't go below zero
        if v_i < 0: v_i = 0

        d_i = prev.distance + (v_i * delta_t) + (0.5 * a_i * (delta_t ** 2))

        # 2. Run your mathematical formulations
        # Note: Efficiency formulas often expect km/h
        v_kmh = v_i * 3.6
        eta, Pi = self.calculate_total_efficiency(v_kmh, a_i, xi_ax, xi_n)

        # 3. Calculate fuel consumption for this instant
        # We pass your required parameters to your specific function
        # Here we use Pi * delta_t as Energy 1 and 0 as Energy 2 (placeholder for braking/others)
        instant_fuel = self.fuel_consumption(Pi * delta_t, 0, self.p['HL'] * eta if eta > 0 else 1)
        if eta <= 0 or Pi <= 0: instant_fuel = 0

        # 4. Store the return values
        record = StateRecord(
            time_index=prev.time_index + delta_t,
            delta_t=delta_t,
            acceleration=a_i,
            velocity=v_i,
            distance=d_i,
            efficiency=eta,
            power=Pi,
            fuel_consumed=instant_fuel
        )

        self.history.append(record)
        return record

    def calculate_ni(self, v, xi_ax, xi_n):
        # equation (9) -
        return (9.55 * v * xi_ax * xi_n) / self.p['rd']

    def calculate_Pe(self, ni):
        # equation (10)
        ratio = ni / self.p['nP']
        termos = (self.p['a'] * ratio +
                  self.p['b'] * ratio ** 2 -
                  self.p['c'] * ratio ** 3)
        return self.p['PP'] * termos

    def calculate_Pi(self, v, a):
        # Equação (11) - Potência instantânea necessária
        v_ms = v / 3.6  # Conversão de km/h para m/s se necessário

        termo_rolamento = self.p['mv'] * self.p['g'] * self.p['fr'] * v_ms
        termo_aerodinamico = 0.5 * self.p['rho_a'] * self.p['CD'] * self.p['Af'] * (v_ms ** 3)
        termo_inercial = self.p['mv'] * self.p['gamma_m'] * a * v_ms

        return (1 / self.p['eta_T']) * (termo_rolamento + termo_aerodinamico + termo_inercial)

    def calculate_total_efficiency(self, v, a, xi_ax, xi_n):
        ni = self.calculate_ni(v, xi_ax, xi_n)
        Pe = self.calculate_Pe(ni)
        Pi = self.calculate_Pi(v, a)

        # Razões para os polinómios
        r_P = Pi / Pe
        r_n = ni / self.p['nP']

        # Equação (12) - mu_P
        mu_P = 0.5968 - 0.1666 * r_P + 2.4968 * (r_P ** 2) - 2.1128 * (r_P ** 3)

        # Equação (13) - mu_n
        mu_n = 0.7107 + 0.9963 * r_n - 1.0582 * (r_n ** 2) + 0.3124 * (r_n ** 3)

        # Equação (14) - Eficiência final
        eta = self.p['eta_o'] * mu_P * mu_n
        return eta, Pi

    def equation_E1(self):
        # power consumption going to overcome resistance
        equation_E1 = (1 / eta_t) * (1 / eta_i) * (m_v * g * f_r * (v_i / 3.6))
        return equation_E1

    def equation_E2(self):
        # power consumption to change kinetic energy
        equation_E2 = ((m_v * lambda_m) / 2) * (a_i / eta_i) * delta_t
        return equation_E2


    def fuel_consumption(self, E_1, E_2, H_L):
        return (E_1 + E_2) / H_L

