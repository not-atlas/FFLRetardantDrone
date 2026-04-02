#!/usr/bin/env python3
"""
Conceptual Sizing Tool for a Fixed-Wing VTOL Firefighting UAV
==============================================================

Wing sizing is driven by stall constraints.
Cruise performance is evaluated from resulting geometry.
Battery sizing is driven by both mission energy and peak power.
Iterative mass convergence resolves weight loops.

Output relies purely on multi-dimensional tables (Pandas DataFrames).
Contains raw generated data points mapped across variable sweeps.
"""

import copy
import itertools
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
g = 9.81  # m/s²

# ═══════════════════════════════════════════════════════════════════════════
# INPUT DATA-CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MissionInputs:
    payload_mass_kg: float = 20.0
    n_payload_units: int = 1
    outbound_range_m: float = 15_000.0      # 15 km
    return_range_m: float = 15_000.0
    cruise_speed_mps: float = 30.0
    target_stall_speed_mps: float = 18.0
    cruise_altitude_density_kgpm3: float = 1.225
    hover_time_takeoff_s: float = 30.0
    hover_time_landing_s: float = 30.0
    transition_time_out_s: float = 15.0
    transition_time_in_s: float = 15.0
    attack_or_release_time_s: float = 30.0
    retry_count: int = 1
    reserve_fraction: float = 0.20
    usable_battery_fraction: float = 0.80

@dataclass
class AircraftInputs:
    mtow_guess_kg: float = 45.0
    margin_fraction: float = 0.10

@dataclass
class AeroInputs:
    cl_max: float = 1.4
    cl_cr_target: float = 0.5           # Used for optional comparison against stall sizing
    aspect_ratio: float = 10.0
    taper_ratio: float = 0.45
    leading_edge_sweep_deg: float = 2.0
    dihedral_deg: float = 3.0
    thickness_to_chord: float = 0.12
    cd0: float = 0.045                  # Adjusted to 0.045
    oswald_efficiency: float = 0.75     # Adjusted to 0.75
    dynamic_viscosity_kgpms: float = 1.789e-5

@dataclass
class VTOLInputs:
    n_lift_rotors: int = 4
    lift_rotor_diameter_m: float = 0.60
    hover_figure_of_merit: float = 0.65
    hover_thrust_margin_factor: float = 1.30
    vertical_climb_rate_mps: float = 3.0
    transition_power_factor: float = 0.85

@dataclass
class CruisePropulsionInputs:
    propulsive_efficiency_cruise: float = 0.70
    electrical_efficiency_total: float = 0.85

@dataclass
class BatteryInputs:
    battery_specific_energy_Whpkg: float = 200.0
    battery_nominal_voltage_V: float = 44.4        # 12S LiPo
    battery_min_voltage_under_load_V: float = 38.4

@dataclass
class MassModelInputs:
    avionics_mass_kg: float = 1.5
    release_mechanism_mass_kg: float = 0.8
    landing_gear_or_skid_mass_kg: float = 0.6
    wiring_misc_mass_kg: float = 0.7
    fuselage_mass_guess_kg: float = 3.5
    wing_areal_density_kgpm2: float = 3.5
    tail_areal_density_kgpm2: float = 2.5
    boom_linear_density_kgpm: float = 0.30
    n_booms: int = 2
    boom_length_m: float = 1.0
    motor_specific_power_Wpkg: float = 5_000.0
    esc_specific_power_Wpkg: float = 8_000.0
    lift_prop_mass_each_kg: float = 0.10
    cruise_prop_mass_kg: float = 0.15
    tail_volume_coefficient_horizontal: float = 0.50
    tail_volume_coefficient_vertical: float = 0.04
    horizontal_tail_arm_m: float = 1.1
    vertical_tail_arm_m: float = 1.1

@dataclass
class ConvergenceSettings:
    tolerance: float = 0.005          # 0.5 % MTOW error
    max_iterations: int = 80
    relaxation: float = 0.40          # blend factor (0..1] – lower = stabler


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS DATA-CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SizingResult:
    converged: bool = False
    iterations: int = 0
    mtow_calc_kg: float = 0.0
    mtow_error_fraction: float = 0.0

    # Geometric & Aerodynamic
    wing_area_stall_req_m2: float = 0.0
    wing_area_cruise_req_m2: float = 0.0
    wing_area_actual_m2: float = 0.0
    wingspan_m: float = 0.0
    root_chord_m: float = 0.0
    tip_chord_m: float = 0.0
    mac_m: float = 0.0

    # Cruise aero outputs
    cruise_dynamic_pressure_Pa: float = 0.0
    cruise_cl_actual: float = 0.0
    cruise_cd_induced: float = 0.0
    cruise_cd_total: float = 0.0
    cruise_drag_N: float = 0.0
    cruise_power_aero_W: float = 0.0
    cruise_power_elec_W: float = 0.0

    # VTOL
    vtol_area_per_rotor_m2: float = 0.0
    vtol_total_disk_area_m2: float = 0.0
    vtol_disk_loading_Npm2: float = 0.0
    hover_induced_velocity_mps: float = 0.0
    hover_power_ideal_W: float = 0.0
    hover_power_elec_W: float = 0.0
    transition_power_elec_W: float = 0.0
    attack_power_elec_W: float = 0.0

    # Mission Energy Tracking
    energy_hover_takeoff_Wh: float = 0.0
    energy_hover_landing_Wh: float = 0.0
    energy_transition_out_Wh: float = 0.0
    energy_transition_in_Wh: float = 0.0
    energy_outbound_cruise_Wh: float = 0.0
    energy_return_cruise_Wh: float = 0.0
    energy_attack_Wh: float = 0.0
    energy_retries_Wh: float = 0.0
    total_mission_energy_Wh: float = 0.0
    required_usable_energy_Wh: float = 0.0
    required_nominal_energy_Wh: float = 0.0

    # Battery
    battery_mass_kg: float = 0.0
    battery_capacity_Ah: float = 0.0
    peak_electrical_power_W: float = 0.0
    peak_current_A: float = 0.0
    battery_c_rate: float = 0.0

    # Sub-component mass breakdown
    wing_mass_kg: float = 0.0
    tail_mass_kg: float = 0.0
    booms_mass_kg: float = 0.0
    lift_motor_mass_kg: float = 0.0
    cruise_motor_mass_kg: float = 0.0
    esc_mass_kg: float = 0.0
    props_mass_kg: float = 0.0
    systems_mass_kg: float = 0.0
    structure_mass_kg: float = 0.0
    margin_mass_kg: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# CORE SIZING LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def _size_once(
    mtow_kg: float,
    mission: MissionInputs,
    aircraft: AircraftInputs,
    aero: AeroInputs,
    vtol: VTOLInputs,
    cruise_prop: CruisePropulsionInputs,
    batt: BatteryInputs,
    mass_model: MassModelInputs,
) -> SizingResult:
    """Run sequential sizing relations for a guessed MTOW."""
    r = SizingResult()
    rho = mission.cruise_altitude_density_kgpm3
    weight_N = mtow_kg * g

    # 1. Wing Sizing
    r.wing_area_stall_req_m2 =  (2 * weight_N) / (rho * mission.target_stall_speed_mps**2 * aero.cl_max)
    r.wing_area_cruise_req_m2 = (2 * weight_N) / (rho * mission.cruise_speed_mps**2 * aero.cl_cr_target)
    r.wing_area_actual_m2 = max(r.wing_area_stall_req_m2, r.wing_area_cruise_req_m2)

    # Wing Geometry derivations
    r.wingspan_m = np.sqrt(aero.aspect_ratio * r.wing_area_actual_m2)
    r.root_chord_m = 2 * r.wing_area_actual_m2 / (r.wingspan_m * (1 + aero.taper_ratio))
    r.tip_chord_m = aero.taper_ratio * r.root_chord_m
    r.mac_m = ((2 / 3) * r.root_chord_m * (1 + aero.taper_ratio + aero.taper_ratio**2) / (1 + aero.taper_ratio))

    # 2. Cruise Evaluation (Resulting vs assumed)
    r.cruise_dynamic_pressure_Pa = 0.5 * rho * mission.cruise_speed_mps**2
    r.cruise_cl_actual = weight_N / (r.cruise_dynamic_pressure_Pa * r.wing_area_actual_m2)
    r.cruise_cd_induced = r.cruise_cl_actual**2 / (np.pi * aero.oswald_efficiency * aero.aspect_ratio)
    r.cruise_cd_total = aero.cd0 + r.cruise_cd_induced
    
    r.cruise_drag_N = r.cruise_dynamic_pressure_Pa * r.wing_area_actual_m2 * r.cruise_cd_total
    r.cruise_power_aero_W = r.cruise_drag_N * mission.cruise_speed_mps
    r.cruise_power_elec_W = r.cruise_power_aero_W / (cruise_prop.propulsive_efficiency_cruise * cruise_prop.electrical_efficiency_total)

    # 3. VTOL Model
    r.vtol_area_per_rotor_m2 = np.pi * vtol.lift_rotor_diameter_m**2 / 4
    r.vtol_total_disk_area_m2 = vtol.n_lift_rotors * r.vtol_area_per_rotor_m2
    r.vtol_disk_loading_Npm2 = weight_N / r.vtol_total_disk_area_m2
    
    thrust_hover_req_N = vtol.hover_thrust_margin_factor * weight_N
    
    # Using momentum theory bounds
    r.hover_induced_velocity_mps = np.sqrt(weight_N / (2 * rho * r.vtol_total_disk_area_m2))
    r.hover_power_ideal_W = weight_N**1.5 / np.sqrt(2 * rho * r.vtol_total_disk_area_m2)
    r.hover_power_elec_W = r.hover_power_ideal_W / (vtol.hover_figure_of_merit * cruise_prop.electrical_efficiency_total)
    
    r.transition_power_elec_W = vtol.transition_power_factor * r.hover_power_elec_W
    r.attack_power_elec_W = r.cruise_power_elec_W

    # 4. Mission Energy Profiling
    outbound_cruise_time_s = mission.outbound_range_m / mission.cruise_speed_mps
    return_cruise_time_s = mission.return_range_m / mission.cruise_speed_mps

    r.energy_hover_takeoff_Wh = r.hover_power_elec_W * mission.hover_time_takeoff_s / 3600
    r.energy_hover_landing_Wh = r.hover_power_elec_W * mission.hover_time_landing_s / 3600
    r.energy_transition_out_Wh = r.transition_power_elec_W * mission.transition_time_out_s / 3600
    r.energy_transition_in_Wh = r.transition_power_elec_W * mission.transition_time_in_s / 3600
    r.energy_outbound_cruise_Wh = r.cruise_power_elec_W * outbound_cruise_time_s / 3600
    r.energy_return_cruise_Wh = r.cruise_power_elec_W * return_cruise_time_s / 3600
    r.energy_attack_Wh = r.attack_power_elec_W * mission.attack_or_release_time_s / 3600
    r.energy_retries_Wh = mission.retry_count * (r.energy_hover_takeoff_Wh + r.energy_transition_out_Wh)

    r.total_mission_energy_Wh = (
        r.energy_hover_takeoff_Wh + r.energy_transition_out_Wh +
        r.energy_outbound_cruise_Wh + r.energy_attack_Wh +
        r.energy_return_cruise_Wh + r.energy_transition_in_Wh +
        r.energy_hover_landing_Wh + r.energy_retries_Wh
    )

    r.required_usable_energy_Wh = r.total_mission_energy_Wh * (1 + mission.reserve_fraction)
    r.required_nominal_energy_Wh = r.required_usable_energy_Wh / mission.usable_battery_fraction

    # 5. Battery Mapping
    r.battery_mass_kg = r.required_nominal_energy_Wh / batt.battery_specific_energy_Whpkg
    r.battery_capacity_Ah = r.required_nominal_energy_Wh / batt.battery_nominal_voltage_V

    r.peak_electrical_power_W = max(r.cruise_power_elec_W, r.hover_power_elec_W, r.transition_power_elec_W, r.attack_power_elec_W)
    r.peak_current_A = r.peak_electrical_power_W / batt.battery_min_voltage_under_load_V
    r.battery_c_rate = r.peak_current_A / r.battery_capacity_Ah if r.battery_capacity_Ah > 0 else 0.0

    # 6. Mass breakdown derivation formulas
    h_tail_area_m2 = (mass_model.tail_volume_coefficient_horizontal * r.wing_area_actual_m2 * r.mac_m / mass_model.horizontal_tail_arm_m)
    v_tail_area_m2 = (mass_model.tail_volume_coefficient_vertical * r.wing_area_actual_m2 * r.wingspan_m / mass_model.vertical_tail_arm_m)

    r.wing_mass_kg = mass_model.wing_areal_density_kgpm2 * r.wing_area_actual_m2
    r.tail_mass_kg = mass_model.tail_areal_density_kgpm2 * (h_tail_area_m2 + v_tail_area_m2)
    r.booms_mass_kg = (mass_model.n_booms * mass_model.boom_linear_density_kgpm * mass_model.boom_length_m)

    vtol_lift_peak_sys_W = r.hover_power_elec_W * vtol.hover_thrust_margin_factor
    r.lift_motor_mass_kg = vtol_lift_peak_sys_W / mass_model.motor_specific_power_Wpkg
    r.cruise_motor_mass_kg = r.cruise_power_elec_W / mass_model.motor_specific_power_Wpkg
    r.esc_mass_kg = r.peak_electrical_power_W / mass_model.esc_specific_power_Wpkg
    r.props_mass_kg = (vtol.n_lift_rotors * mass_model.lift_prop_mass_each_kg + mass_model.cruise_prop_mass_kg)

    r.systems_mass_kg = (mass_model.avionics_mass_kg + mass_model.release_mechanism_mass_kg +
                         mass_model.landing_gear_or_skid_mass_kg + mass_model.wiring_misc_mass_kg)
    r.structure_mass_kg = r.wing_mass_kg + r.tail_mass_kg + r.booms_mass_kg + mass_model.fuselage_mass_guess_kg

    payload_total_kg = mission.payload_mass_kg * mission.n_payload_units
    subtotal_mass_kg = (payload_total_kg + r.battery_mass_kg + r.lift_motor_mass_kg + 
                        r.cruise_motor_mass_kg + r.esc_mass_kg + r.props_mass_kg + 
                        r.systems_mass_kg + r.structure_mass_kg)

    r.margin_mass_kg = aircraft.margin_fraction * subtotal_mass_kg
    
    r.mtow_calc_kg = subtotal_mass_kg + r.margin_mass_kg
    r.mtow_error_fraction = ((r.mtow_calc_kg - mtow_kg) / mtow_kg if mtow_kg > 0 else 0)

    return r

def size_aircraft(
    mission: MissionInputs,
    aircraft: AircraftInputs,
    aero: AeroInputs,
    vtol: VTOLInputs,
    cruise_prop: CruisePropulsionInputs,
    batt: BatteryInputs,
    mass_model: MassModelInputs,
    convergence: ConvergenceSettings = ConvergenceSettings(),
) -> SizingResult:
    """Recursively sizes UAV masses against assumed initial states until fixed."""
    mtow = aircraft.mtow_guess_kg
    alpha = convergence.relaxation

    for i in range(1, convergence.max_iterations + 1):
        r = _size_once(mtow, mission, aircraft, aero, vtol, cruise_prop, batt, mass_model)
        mtow_new = r.mtow_calc_kg
        mtow = mtow * (1 - alpha) + mtow_new * alpha
        
        if abs(r.mtow_error_fraction) < convergence.tolerance:
            r.converged = True
            r.iterations = i
            # Final rigid solve
            return _size_once(mtow, mission, aircraft, aero, vtol, cruise_prop, batt, mass_model)
            
    # Max iter exceeded
    r = _size_once(mtow, mission, aircraft, aero, vtol, cruise_prop, batt, mass_model)
    r.converged = False
    r.iterations = convergence.max_iterations
    return r

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS ENGINE - RAW DATA MAPPER
# ═══════════════════════════════════════════════════════════════════════════

def flatten_inputs(inputs: dict) -> dict:
    """Takes nested dataclasses and flattens them for DataFrame rows."""
    flattened = {}
    for group_name, dtcls_obj in inputs.items():
        if group_name == 'convergence':
            continue
        for field_name, value in asdict(dtcls_obj).items():
            flattened[f"INP_{group_name}_{field_name}"] = value
    return flattened

def parameter_sweep(
    sweep_params: Dict[str, np.ndarray],
    base_inputs: Dict[str, object],
) -> pd.DataFrame:
    """
    Run N-D grid sweeps over inputs and map flat Outputs to Input matrices.
    Returns complete multi-dimensional datatable.
    """
    group_map = {
        "mission": "mission", "aircraft": "aircraft", "aero": "aero",
        "vtol": "vtol", "cruise_prop": "cruise_prop", "batt": "batt",
        "mass_model": "mass_model",
    }

    param_names = list(sweep_params.keys())
    param_arrays = [sweep_params[k] for k in param_names]
    grid = list(itertools.product(*param_arrays))

    records = []
    
    for combo in grid:
        inputs = {k: copy.deepcopy(v) for k, v in base_inputs.items()}
        for pname, val in zip(param_names, combo):
            grp, fld = pname.split(".", 1)
            setattr(inputs[group_map[grp]], fld, val)

        # Execute sizing iteration
        r = size_aircraft(
            inputs["mission"], inputs["aircraft"], inputs["aero"],
            inputs["vtol"], inputs["cruise_prop"], inputs["batt"],
            inputs["mass_model"], inputs.get("convergence", ConvergenceSettings()),
        )

        row = flatten_inputs(inputs)
        
        # Append calculated outputs uniformly
        for field_name, value in asdict(r).items():
            row[f"OUT_{field_name}"] = value
            
        records.append(row)

    return pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════════════════
# GRAPHING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def generate_sweep_graphs(df: pd.DataFrame):
    """Generates graphs plotting the sweeps of input variables vs energy/wing outputs."""
    
    # Example 1: Cruise Speed vs Wing Area & Mission Energy (Color by Payload)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    payloads = sorted(df["INP_mission_payload_mass_kg"].unique())
    
    unique_d = sorted(df["INP_vtol_lift_rotor_diameter_m"].unique())
    fixes_d = unique_d[len(unique_d) // 2]
    df_fixed_d = df[df["INP_vtol_lift_rotor_diameter_m"] == fixes_d]
    
    for p in payloads:
        sub_df = df_fixed_d[df_fixed_d["INP_mission_payload_mass_kg"] == p]
        axes[0].plot(sub_df["INP_mission_cruise_speed_mps"], sub_df["OUT_wing_area_actual_m2"], marker="o", label=f"Payload: {p}kg")
        axes[1].plot(sub_df["INP_mission_cruise_speed_mps"], sub_df["OUT_total_mission_energy_Wh"], marker="s", label=f"Payload: {p}kg")
        
    axes[0].set_title("Cruise Speed vs Wing Area")
    axes[0].set_xlabel("Cruise Speed (m/s)")
    axes[0].set_ylabel("Wing Area (m²)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title("Cruise Speed vs Total Mission Energy")
    axes[1].set_xlabel("Cruise Speed (m/s)")
    axes[1].set_ylabel("Mission Energy (Wh)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    fig.suptitle(f"Sweeps Fixed at Rotor Diameter = {fixes_d:.2f} m")
    fig.tight_layout()
    fig.savefig("sweep_speed_vs_wing_energy.png", dpi=150, bbox_inches="tight")
    
    # Example 2: Rotor Diameter vs Energy Outputs (Hover power dictates mass penalties)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    unique_v = sorted(df["INP_mission_cruise_speed_mps"].unique())
    fixes_v = unique_v[len(unique_v) // 2]
    df_fixed_v = df[df["INP_mission_cruise_speed_mps"] == fixes_v]
    
    for p in payloads:
        sub_df = df_fixed_v[df_fixed_v["INP_mission_payload_mass_kg"] == p]
        axes2[0].plot(sub_df["INP_vtol_lift_rotor_diameter_m"], sub_df["OUT_total_mission_energy_Wh"], marker="o", label=f"Payload: {p}kg")
        axes2[1].plot(sub_df["INP_vtol_lift_rotor_diameter_m"], sub_df["OUT_battery_mass_kg"], marker="s", label=f"Payload: {p}kg")

    axes2[0].set_title("Rotor Diameter vs Total Mission Energy")
    axes2[0].set_xlabel("Rotor Diameter (m)")
    axes2[0].set_ylabel("Total Mission Energy (Wh)")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()
    
    axes2[1].set_title("Rotor Diameter vs Battery Mass")
    axes2[1].set_xlabel("Rotor Diameter (m)")
    axes2[1].set_ylabel("Battery Mass (kg)")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    fig2.suptitle(f"Sweeps Fixed at Cruise Speed = {fixes_v:.2f} m/s")
    fig2.tight_layout()
    fig2.savefig("sweep_rotor_vs_battery_power.png", dpi=150, bbox_inches="tight")
    print("  -> Saved 'sweep_speed_vs_wing_energy.png' and 'sweep_rotor_vs_battery_power.png'")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN 
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  VTOL Firefighting UAV - Multidimensional Table Engine")
    print("=" * 72)

    # Initialize Base Configurations
    base_inputs = dict(
        mission=MissionInputs(),
        aircraft=AircraftInputs(),
        aero=AeroInputs(),
        vtol=VTOLInputs(),
        cruise_prop=CruisePropulsionInputs(),
        batt=BatteryInputs(),
        mass_model=MassModelInputs(),
        convergence=ConvergenceSettings()
    )

    # Setup ranges across 3 independent axes
    sweep_params = {
        "mission.payload_mass_kg": np.array([4.54, 6.8]),
        "mission.cruise_speed_mps": np.arange(45.0, 51.0, 1.0),
        "vtol.lift_rotor_diameter_m": np.arange(0.20, 0.36, 0.05)
    }

    print(f"\n[!] Generating grid. Total permutations scheduled: {np.prod([len(v) for v in sweep_params.values()])}")
    
    # Process the dataframe matrix
    df_results = parameter_sweep(sweep_params, base_inputs)
    
    csv_filename = "vtol_sizing_results.csv"
    df_results.to_csv(csv_filename, index=False)
    
    print(f"\n✓ Exported highly-dimensional table to: {csv_filename}")
    print(f"  Shape: {df_results.shape[0]} rows × {df_results.shape[1]} columns")

    # Present a preview of some critical dimensions mapping Inputs -> Selected Outputs
    print("\n[Preview] Selected cross-section from the results table:")
    
    # Selecting limited dimensions for clean console display
    preview_cols = [
        "INP_mission_payload_mass_kg",
        "INP_mission_cruise_speed_mps", 
        "INP_vtol_lift_rotor_diameter_m",
        "OUT_mtow_calc_kg",
        "OUT_wing_area_actual_m2",
        "OUT_cruise_cl_actual",
        "OUT_battery_mass_kg",
        "OUT_total_mission_energy_Wh",
        "OUT_battery_c_rate",
        "OUT_converged"
    ]
    
    preview_df = df_results[preview_cols].copy()
    
    # Rename for console legibility only
    prettier_names = {
        "INP_mission_payload_mass_kg": "Payload_kg",
        "INP_mission_cruise_speed_mps": "Speed_mps",
        "INP_vtol_lift_rotor_diameter_m": "Rotor_D_m",
        "OUT_mtow_calc_kg": "MTOW_kg",
        "OUT_wing_area_actual_m2": "Wing_Area",
        "OUT_cruise_cl_actual": "Actual_CL_cr",
        "OUT_battery_mass_kg": "Battery_kg",
        "OUT_total_mission_energy_Wh": "Energy_Wh",
        "OUT_battery_c_rate": "C_rate",
        "OUT_converged": "Converged"
    }
    preview_df.rename(columns=prettier_names, inplace=True)
    
    print(preview_df.head(15).to_string(index=False, float_format="%.2f"))

    # Generate graphs of Input Variables vs Output Variables mapping Energy / Wing logic
    print("\n[!] Generating graphs of sweeps (Inputs vs Wing Sizing / Energy)...")
    generate_sweep_graphs(df_results)
    
if __name__ == "__main__":
    main()
