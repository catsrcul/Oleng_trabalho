#### Super summary:

This project optimizes fuel use by finding the best speed profile.
The model combines physics with engine maps to predict energy cost.
Vehicle constants (mass, aerodynamics, and engine curves) define the car.
The output is fuel used over a given distance to evaluate a driving strategy.
The discretization uses distance steps to match road segments.


# Calculations
There is a problem in the formulation (formula 12 in the study):

$$ P_i = \frac{1}{\eta_T} \left( m_v \cdot g \cdot f_r \cdot \frac{v(t)}{3.6} + 0.5 \cdot \rho_a \cdot C_D \cdot A_f \cdot \left\{ \frac{v(t)}{3.6} \right\}^3 + m_v \cdot \gamma_m \cdot a(t) \right) $$
Where, in the final element for inertia, there is a unit problem and the correct formula should be:
$$ P_i = \frac{1}{\eta_T} \left( m_v g f_r + 0.5 \rho_a C_D A_f v^2 + m_v \gamma_m a \right) \cdot v $$
In the previous equations we did not sum power but force for the inertia.
$$ \text{Units} = [Mass] \cdot [Acceleration] = \textcolor{red}{\text{Newtons}} \text{ (Force)}$$
