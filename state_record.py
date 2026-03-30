from dataclasses import dataclass

@dataclass
class StateRecord:
    """
    Represents a snapshot of the vehicle state at a specific moment.
    """

    time_index: float
    delta_t: float              # diference in time from last index
    acceleration: float
    velocity: float
    distance: float
    efficiency: float
    power: float
    fuel_consumed: float