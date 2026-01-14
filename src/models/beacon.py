from dataclasses import dataclass
import numpy as np

@dataclass
class UWBHardwareParams:
    # Power consumption (mW)
    P_COR: float = 10.0
    P_ADC: float = 15.0
    P_LNA: float = 9.0
    P_VGA: float = 5.0
    P_GEN: float = 8.0
    P_SYN: float = 6.0
    P_EST: float = 7.0

    # Timing parameters (seconds)
    T_SP: float = 0.0004
    T_PHR: float = 0.0002
    T_PAYLOAD: float = 0.002
    T_TR: float = 0.0001
    T_IPS: float = 0.00005
    T_ACK: float = 0.0002

    # Control flags
    rho_c: int = 1   # soft decision
    rho_r: int = 1   # coherent receiver
    M: int = 1       # number of correlator branches


class UWBEnergyModel:
    def __init__(self, params: UWBHardwareParams):
        self.p = params
        self.energy_per_packet = self._compute_energy_per_packet()

    def _receiver_power(self):
        Pd = (
            self.p.M * self.p.P_COR +
            self.p.rho_c * self.p.P_ADC +
            self.p.P_LNA +
            self.p.P_VGA
        )
        Pn = self.p.rho_r * (self.p.P_GEN + self.p.P_SYN + self.p.P_EST)
        return Pd + Pn

    def _compute_energy_per_packet(self):
        Pr = self._receiver_power()

        # Overhead energy (SP + PHR)
        EO = Pr * (self.p.T_SP + self.p.T_PHR)

        # Payload energy
        EL = Pr * self.p.T_PAYLOAD

        # RX energy
        ERX = EO + EL

        # Transition energy
        Etr = self.p.rho_r * self.p.P_SYN * self.p.T_TR

        # Inter-packet spacing energy
        EIPS = self.p.rho_r * self.p.P_SYN * self.p.T_IPS

        # ACK listening energy
        EACK = self.p.P_SYN * self.p.T_ACK

        # Total energy per localization step
        return 2 * Etr + ERX + 2 * EIPS + EACK


class BeaconBattery:
    def __init__(self, beacon_id, initial_battery, energy_per_use):
        self.id = beacon_id
        self.battery = float(initial_battery)
        self.energy = float(energy_per_use)

    def consume_energy(self):
        """
        Call this when the beacon is selected for localization
        """
        self.battery -= self.energy

    def is_depleted(self):
        return self.battery <= 0

    def get_battery_level(self):
        return self.battery


class Beacon:
    def __init__(self, beacon_id, position, initial_battery, uwb_params):
        self.id = beacon_id
        self.position = np.array(position)
        self.energy_model = UWBEnergyModel(uwb_params)
        self.battery = BeaconBattery(
            beacon_id,
            initial_battery,
            self.energy_model.energy_per_packet
        )

    def energy_per_use(self):
        return self.energy_model.energy_per_packet
    
    def current_battery_level(self):
        return self.battery.get_battery_level()
    
    def is_battery_depleted(self):
        return self.battery.is_depleted()
    
    def position(self):
        return self.position
    
    def beacon_id(self):
        return self.id
    
    def use_for_localization(self):
        self.battery.consume_energy()