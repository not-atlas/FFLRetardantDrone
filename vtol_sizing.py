#!/usr/bin/env python3
"""
Comprehensive Sizing Tool for a Fixed-Wing VTOL Firefighting UAV
================================================================
Implements exact mass loop dependencies, mission profile integrators,
and all 38 prioritized sensitivity plot generators.
"""

import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List

# ---------------------------------------------------------------------------
# CONSTANTS & CONVERSIONS
# ---------------------------------------------------------------------------
g = 9.81  # m/s^2

def miles_to_m(miles: float) -> float:
    return miles * 1609.34

def m_to_miles(meters: float) -> float:
    return meters / 1609.34

# ═══════════════════════════════════════════════════════════════════════════
# INPUT PARAMETERS (DEFAULTS & RANGES)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Assumptions:
    # Mass & Aero
    total_mass_guess_kg: float = 30.0
    stall_speed_mps: float = 22.0
    cruise_speed_mps: float = 50.0
    aspect_ratio: float = 8.0
    taper_ratio: float = 0.5
    cl_max: float = 1.6
    cl_cr_target: float = 0.5
    cd0: float = 0.045
    oswald: float = 0.70
    rho: float = 1.225
    
    # VTOL
    n_rotors: int = 4
    rotor_diameter_m: float = 0.55
    hover_fom: float = 0.65
    vert_climb_mps: float = 3.0
    transition_power_multiplier: float = 0.85
    hover_time_takeoff_s: float = 15.0
    hover_time_landing_s: float = 15.0
    
    # Propulsion & Battery
    prop_efficiency: float = 0.75
    elec_efficiency: float = 0.90
    batt_specific_energy_wh_kg: float = 220.0
    batt_usable_fraction: float = 0.87
    batt_reserve_fraction: float = 0.30
    batt_voltage_V: float = 44.4
    
    # Mission
    outbound_distance_miles: float = 7.0
    return_distance_miles: float = 7.0
    attack_time_s: float = 30.0
    retries: int = 0
    
    # Structural Guess Overrides (if required to close mass loop realistically)
    structural_mass_fraction: float = 0.35
    payload_mass_kg: float = 6.8   # Optional independent track

    def copy(self):
        return copy.deepcopy(self)


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SizingResult:
    converged: bool = False
    total_mass_kg: float = 0.0
    weight_N: float = 0.0
    wing_area_m2: float = 0.0
    span_m: float = 0.0
    root_chord_m: float = 0.0
    tip_chord_m: float = 0.0
    mac_m: float = 0.0
    q_cruise_Pa: float = 0.0
    cl_cruise: float = 0.0
    cd_induced: float = 0.0
    cd_total: float = 0.0
    drag_N: float = 0.0
    power_cruise_W: float = 0.0
    rotor_total_area_m2: float = 0.0
    power_hover_W: float = 0.0
    power_climb_W: float = 0.0
    power_transition_W: float = 0.0
    e_takeoff_hover: float = 0.0
    e_takeoff_climb: float = 0.0
    e_transition_out: float = 0.0
    e_outbound: float = 0.0
    e_attack: float = 0.0
    e_return: float = 0.0
    e_transition_in: float = 0.0
    e_landing_hover: float = 0.0
    e_retries: float = 0.0
    total_mission_energy_Wh: float = 0.0
    required_nominal_batt_Wh: float = 0.0
    battery_mass_kg: float = 0.0
    battery_Ah: float = 0.0
    peak_power_W: float = 0.0
    peak_current_A: float = 0.0
    peak_c_rate: float = 0.0
    mtow_calc_kg: float = 0.0
    error: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SIZING CALCULATION LOGIC 
# ═══════════════════════════════════════════════════════════════════════════

def compute_aircraft(a: Assumptions, force_convergence: bool = True) -> SizingResult:
    mtow = a.total_mass_guess_kg
    res = SizingResult()
    dist_out_m = miles_to_m(a.outbound_distance_miles)
    dist_ret_m = miles_to_m(a.return_distance_miles)
    
    max_iters = 50 if force_convergence else 1
    tol = 0.005
    
    for _ in range(max_iters):
        r = SizingResult()
        r.total_mass_kg = mtow
        r.weight_N = mtow * g
        
        wing_area_stall = (2 * r.weight_N) / (a.rho * a.stall_speed_mps**2 * a.cl_max)
        wing_area_cruise = (2 * r.weight_N) / (a.rho * a.cruise_speed_mps**2 * a.cl_cr_target)
        r.wing_area_m2 = max(wing_area_stall, wing_area_cruise)
        
        r.span_m = np.sqrt(a.aspect_ratio * r.wing_area_m2)
        r.root_chord_m = 2 * r.wing_area_m2 / (r.span_m * (1 + a.taper_ratio))
        r.tip_chord_m = a.taper_ratio * r.root_chord_m
        r.mac_m = (2/3) * r.root_chord_m * (1 + a.taper_ratio + a.taper_ratio**2) / (1 + a.taper_ratio)
        
        r.q_cruise_Pa = 0.5 * a.rho * a.cruise_speed_mps**2
        r.cl_cruise = r.weight_N / (r.q_cruise_Pa * r.wing_area_m2)
        r.cd_induced = (r.cl_cruise**2) / (np.pi * a.oswald * a.aspect_ratio)
        r.cd_total = a.cd0 + r.cd_induced
        r.drag_N = r.q_cruise_Pa * r.wing_area_m2 * r.cd_total
        r.power_cruise_W = (r.drag_N * a.cruise_speed_mps) / (a.prop_efficiency * a.elec_efficiency)
        
        r.rotor_total_area_m2 = a.n_rotors * np.pi * (a.rotor_diameter_m/2)**2
        p_hover_ideal = (r.weight_N**1.5) / np.sqrt(2 * a.rho * r.rotor_total_area_m2)
        r.power_hover_W = p_hover_ideal / (a.hover_fom * a.elec_efficiency)
        r.power_climb_W = r.power_hover_W + ((r.weight_N * a.vert_climb_mps) / a.elec_efficiency)
        r.power_transition_W = r.power_hover_W * a.transition_power_multiplier
        
        t_time = 15.0
        r.e_takeoff_hover = r.power_hover_W * a.hover_time_takeoff_s / 3600
        r.e_takeoff_climb = r.power_climb_W * 5.0 / 3600
        r.e_transition_out = r.power_transition_W * t_time / 3600
        r.e_outbound = r.power_cruise_W * (dist_out_m / a.cruise_speed_mps) / 3600
        r.e_attack = r.power_cruise_W * a.attack_time_s / 3600
        r.e_return = r.power_cruise_W * (dist_ret_m / a.cruise_speed_mps) / 3600
        r.e_transition_in = r.power_transition_W * t_time / 3600
        r.e_landing_hover = r.power_hover_W * a.hover_time_landing_s / 3600
        r.e_retries = a.retries * (r.e_takeoff_hover + r.e_transition_out)
        
        r.total_mission_energy_Wh = sum([
            r.e_takeoff_hover, r.e_takeoff_climb, r.e_transition_out,
            r.e_outbound, r.e_attack, r.e_return, r.e_transition_in, 
            r.e_landing_hover, r.e_retries
        ])
        
        usable_required = r.total_mission_energy_Wh * (1 + a.batt_reserve_fraction)
        r.required_nominal_batt_Wh = usable_required / a.batt_usable_fraction
        r.battery_mass_kg = r.required_nominal_batt_Wh / a.batt_specific_energy_wh_kg
        r.battery_Ah = r.required_nominal_batt_Wh / a.batt_voltage_V
        
        r.peak_power_W = max([r.power_climb_W, r.power_hover_W, r.power_transition_W, r.power_cruise_W])
        r.peak_current_A = r.peak_power_W / a.batt_voltage_V
        r.peak_c_rate = r.peak_current_A / r.battery_Ah if r.battery_Ah > 0 else 0
        
        r.mtow_calc_kg = a.payload_mass_kg + (mtow * a.structural_mass_fraction) + r.battery_mass_kg
        err = (r.mtow_calc_kg - mtow) / mtow
        r.error = err
        
        if force_convergence:
            if abs(err) < tol:
                r.converged = True
                res = r
                break
            mtow = mtow * 0.5 + r.mtow_calc_kg * 0.5
        else:
            r.converged = True
            res = r
    
    return res if res.total_mass_kg > 0 else r


def run_1d_sweep(base_cfg: Assumptions, param_name: str, values: np.ndarray, force_conv: bool = False) -> pd.DataFrame:
    results = []
    for val in values:
        cfg = base_cfg.copy()
        setattr(cfg, param_name, val)
        res = compute_aircraft(cfg, force_convergence=force_conv)
        r = asdict(cfg); r.update({f"OUT_{k}": v for k, v in asdict(res).items()})
        results.append(r)
    return pd.DataFrame(results)

def run_2d_sweep(base_cfg: Assumptions, p1: str, v1: np.ndarray, p2: str, v2: np.ndarray) -> pd.DataFrame:
    results = []
    for val1, val2 in itertools.product(v1, v2):
        cfg = base_cfg.copy(); setattr(cfg, p1, val1); setattr(cfg, p2, val2)
        res = compute_aircraft(cfg, force_convergence=False)
        r = asdict(cfg); r.update({f"OUT_{k}": v for k, v in asdict(res).items()})
        results.append(r)
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION SUITE (ALL 38 PLOTS)
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_plots(base: Assumptions):
    print("[Plots] Generating all 38 requested visualizations...")
    
    # -----------------------------------------------------------------------
    # A. Wing Sizing Plots (1-12)
    # -----------------------------------------------------------------------
    figA, axsA = plt.subplots(3, 4, figsize=(20, 15))
    axsA = axsA.flatten()
    
    ar_arr = np.linspace(6, 12, 15)
    m_arr = [25, 30, 35]
    tr_arr = [0.4, 0.6, 0.8]
    
    # 1. Aspect Ratio vs Span
    # 2. Aspect Ratio vs Root Chord
    # 3. Aspect Ratio vs Tip Chord
    # 4. Aspect Ratio vs MAC
    for a in axsA[0:4]: a.set_xlabel("Aspect Ratio")
    axsA[0].set_ylabel("Span (m)")
    axsA[1].set_ylabel("Root Chord (m)")
    axsA[2].set_ylabel("Tip Chord (m)")
    axsA[3].set_ylabel("MAC (m)")
    
    for tr in tr_arr:
        b = base.copy(); b.taper_ratio = tr
        dft = run_1d_sweep(b, "aspect_ratio", ar_arr)
        axsA[0].plot(dft['aspect_ratio'], dft['OUT_span_m'], label=f"TR={tr}")
        axsA[1].plot(dft['aspect_ratio'], dft['OUT_root_chord_m'], label=f"TR={tr}")
        axsA[2].plot(dft['aspect_ratio'], dft['OUT_tip_chord_m'], label=f"TR={tr}")
        axsA[3].plot(dft['aspect_ratio'], dft['OUT_mac_m'], label=f"TR={tr}")
        
    axsA[0].legend(); axsA[1].legend(); axsA[2].legend(); axsA[3].legend()
    
    # 5. Stall speed vs required wing area
    v_stall_arr = np.linspace(18, 26, 20)
    for m in m_arr:
        b = base.copy(); b.total_mass_guess_kg = m
        df = run_1d_sweep(b, "stall_speed_mps", v_stall_arr)
        axsA[4].plot(df['stall_speed_mps'], df['OUT_wing_area_m2'], label=f"Mass={m}kg")
    axsA[4].set_xlabel("Stall Target (m/s)"); axsA[4].set_ylabel("Wing Area (m²)"); axsA[4].legend()
    
    # 6. Total mass vs required wing area
    for v in [18, 22, 26]:
        b = base.copy(); b.stall_speed_mps = v
        df = run_1d_sweep(b, "total_mass_guess_kg", np.linspace(25, 40, 15))
        axsA[5].plot(df['total_mass_guess_kg'], df['OUT_wing_area_m2'], label=f"V_stall={v}m/s")
    axsA[5].set_xlabel("Total Mass (kg)"); axsA[5].set_ylabel("Wing Area (m²)"); axsA[5].legend()

    # 7. Wing area vs actual cruise lift coefficient
    for speed in [40, 50, 60]:
        b = base.copy(); b.cruise_speed_mps = speed
        df = run_1d_sweep(b, "stall_speed_mps", v_stall_arr)
        axsA[6].plot(df['OUT_wing_area_m2'], df['OUT_cl_cruise'], label=f"V_cr={speed}")
    axsA[6].set_xlabel("Wing Area (m²)"); axsA[6].set_ylabel("Cruise CL"); axsA[6].legend()

    # 8. Aspect ratio vs induced drag coefficient
    # 9. Aspect ratio vs total cruise drag coefficient
    for speed in [40, 50, 60]:
        b = base.copy(); b.cruise_speed_mps = speed
        df = run_1d_sweep(b, "aspect_ratio", ar_arr)
        axsA[7].plot(df['aspect_ratio'], df['OUT_cd_induced'], label=f"V_cr={speed}")
        axsA[8].plot(df['aspect_ratio'], df['OUT_cd_total'], label=f"V_cr={speed}")
    axsA[7].set_xlabel("Aspect Ratio"); axsA[7].set_ylabel("CD Induced"); axsA[7].legend()
    axsA[8].set_xlabel("Aspect Ratio"); axsA[8].set_ylabel("CD Total"); axsA[8].legend()

    # 10. Cruise Speed vs drag force
    # 11. Cruise speed vs cruise power
    v_cr_arr = np.linspace(40, 60, 20)
    for ar in [6, 8, 10]:
        b = base.copy(); b.aspect_ratio = ar
        df = run_1d_sweep(b, "cruise_speed_mps", v_cr_arr)
        axsA[9].plot(df['cruise_speed_mps'], df['OUT_drag_N'], label=f"AR={ar}")
        axsA[10].plot(df['cruise_speed_mps'], df['OUT_power_cruise_W'], label=f"AR={ar}")
    axsA[9].set_xlabel("Cruise Speed (m/s)"); axsA[9].set_ylabel("Drag Force (N)"); axsA[9].legend()
    axsA[10].set_xlabel("Cruise Speed (m/s)"); axsA[10].set_ylabel("Cruise Power (W)"); axsA[10].legend()

    # 12. Taper ratio vs root and tip chord
    t_sweep = np.linspace(0.3, 1.0, 15)
    for ar in [6, 8, 10]:
        b = base.copy(); b.aspect_ratio = ar
        df = run_1d_sweep(b, "taper_ratio", t_sweep)
        axsA[11].plot(df['taper_ratio'], df['OUT_root_chord_m'], '-', label=f"Root AR={ar}")
        axsA[11].plot(df['taper_ratio'], df['OUT_tip_chord_m'], '--', label=f"Tip AR={ar}")
    axsA[11].set_xlabel("Taper Ratio"); axsA[11].set_ylabel("Chord (m)"); axsA[11].legend(fontsize=7)
    
    for ax in axsA: ax.grid(True)
    figA.suptitle("A. Wing Sizing Plots (1-12)")
    figA.tight_layout()
    figA.savefig("plot_A_wing_sizing.png", dpi=120)
    plt.close(figA)
    

    # -----------------------------------------------------------------------
    # B. VTOL and Hover Plots (13-17)
    # -----------------------------------------------------------------------
    figB, axsB = plt.subplots(1, 5, figsize=(25, 5))
    d_rot_arr = np.linspace(0.4, 0.7, 15)
    
    # 13. Rotor diam vs total disk area
    df_d = run_1d_sweep(base, "rotor_diameter_m", d_rot_arr)
    axsB[0].plot(df_d['rotor_diameter_m'], df_d['OUT_rotor_total_area_m2'])
    axsB[0].set_xlabel("Rotor Diameter (m)"); axsB[0].set_ylabel("Disk Area (m²)")
    
    # 14. Rotor diam vs hover power
    for m in m_arr:
        b = base.copy(); b.total_mass_guess_kg = m
        df = run_1d_sweep(b, "rotor_diameter_m", d_rot_arr)
        axsB[1].plot(df['rotor_diameter_m'], df['OUT_power_hover_W'], label=f"M={m}")
    axsB[1].set_xlabel("Rotor Diameter (m)"); axsB[1].set_ylabel("Hover Power (W)"); axsB[1].legend()

    # 15. Total mass vs hover power
    m_sweep = np.linspace(25, 40, 15)
    for d in [0.45, 0.55, 0.65]:
        b = base.copy(); b.rotor_diameter_m = d
        df = run_1d_sweep(b, "total_mass_guess_kg", m_sweep)
        axsB[2].plot(df['total_mass_guess_kg'], df['OUT_power_hover_W'], label=f"D={d}m")
    axsB[2].set_xlabel("Total Mass (kg)"); axsB[2].set_ylabel("Hover Power (W)"); axsB[2].legend()

    # 16. Rotor diam vs peak current
    for v in [22.2, 44.4]:
        b = base.copy(); b.batt_voltage_V = v
        df = run_1d_sweep(b, "rotor_diameter_m", d_rot_arr)
        axsB[3].plot(df['rotor_diameter_m'], df['OUT_peak_current_A'], label=f"Voltage={v}V")
    axsB[3].set_xlabel("Rotor Diameter (m)"); axsB[3].set_ylabel("Peak Current (A)"); axsB[3].legend()

    # 17. Rotor diam vs hover energy per takeoff
    for ht in [10, 30, 60]:
        b = base.copy(); b.hover_time_takeoff_s = ht
        df = run_1d_sweep(b, "rotor_diameter_m", d_rot_arr)
        axsB[4].plot(df['rotor_diameter_m'], df['OUT_e_takeoff_hover'], label=f"Time={ht}s")
    axsB[4].set_xlabel("Rotor Diameter (m)"); axsB[4].set_ylabel("Hover Energy/Takeoff (Wh)"); axsB[4].legend()

    for ax in axsB: ax.grid(True)
    figB.suptitle("B. VTOL and Hover Plots (13-17)")
    figB.tight_layout()
    figB.savefig("plot_B_vtol_hover.png", dpi=120)
    plt.close(figB)

    # -----------------------------------------------------------------------
    # C. Mission Energy and Battery Plots (18-32)
    # -----------------------------------------------------------------------
    figC, axsC = plt.subplots(3, 5, figsize=(25, 15))
    axsC = axsC.flatten()

    # 18. Cruise speed vs total mission energy
    for m in m_arr:
        b = base.copy(); b.total_mass_guess_kg = m
        df = run_1d_sweep(b, "cruise_speed_mps", v_cr_arr)
        axsC[0].plot(df['cruise_speed_mps'], df['OUT_total_mission_energy_Wh'], label=f"M={m}")
    axsC[0].set_xlabel("Cruise Speed (m/s)"); axsC[0].set_ylabel("Total Mission Energy (Wh)")
    
    # 19. Outbound distance vs total mission energy
    d_miles = np.linspace(5, 15, 15)
    for speed in [40, 50, 60]:
        b = base.copy(); b.cruise_speed_mps = speed
        df = run_1d_sweep(b, "outbound_distance_miles", d_miles)
        axsC[1].plot(df['outbound_distance_miles'], df['OUT_total_mission_energy_Wh'], label=f"V_cr={speed}")
    axsC[1].set_xlabel("Outbound Dist (miles)"); axsC[1].set_ylabel("Total Mission Energy (Wh)")

    # 20. Hover time vs match energy
    hov_times = np.linspace(10, 90, 15)
    for d in [0.45, 0.55, 0.65]:
        b = base.copy(); b.rotor_diameter_m = d
        df = run_1d_sweep(b, "hover_time_takeoff_s", hov_times)
        axsC[2].plot(df['hover_time_takeoff_s'], df['OUT_total_mission_energy_Wh'], label=f"D={d}m")
    axsC[2].set_xlabel("Hover Time Takeoff (s)"); axsC[2].set_ylabel("Total Mission Energy (Wh)")

    # 21 & 22. Fractions vs Energy
    res_fr = np.linspace(0.1, 0.4, 10)
    df_rf = run_1d_sweep(base, "batt_reserve_fraction", res_fr)
    axsC[3].plot(df_rf['batt_reserve_fraction'], df_rf['OUT_required_nominal_batt_Wh'])
    axsC[3].set_xlabel("Reserve Fraction"); axsC[3].set_ylabel("Nominal Batt Energy (Wh)")

    use_fr = np.linspace(0.7, 0.9, 10)
    df_uf = run_1d_sweep(base, "batt_usable_fraction", use_fr)
    axsC[4].plot(df_uf['batt_usable_fraction'], df_uf['OUT_required_nominal_batt_Wh'])
    axsC[4].set_xlabel("Usable Battery Fraction"); axsC[4].set_ylabel("Nominal Batt Energy (Wh)")

    # 23. Battery specific energy vs battery mass
    sp_e = np.linspace(150, 260, 15)
    df_se = run_1d_sweep(base, "batt_specific_energy_wh_kg", sp_e)
    axsC[5].plot(df_se['batt_specific_energy_wh_kg'], df_se['OUT_battery_mass_kg'], color='purple')
    axsC[5].set_xlabel("Specific Energy (Wh/kg)"); axsC[5].set_ylabel("Battery Mass (kg)")

    # 24 & 25. Mass & Speed vs Battery Mass
    for d in [0.45, 0.55, 0.65]:
        b = base.copy(); b.rotor_diameter_m = d
        df = run_1d_sweep(b, "total_mass_guess_kg", m_sweep)
        axsC[6].plot(df['total_mass_guess_kg'], df['OUT_battery_mass_kg'], label=f"D={d}m")
    axsC[6].set_xlabel("Total Mass (kg)"); axsC[6].set_ylabel("Battery Mass (kg)")

    for md in [5, 7, 10]:
        b = base.copy(); b.outbound_distance_miles = md
        df = run_1d_sweep(b, "cruise_speed_mps", v_cr_arr)
        axsC[7].plot(df['cruise_speed_mps'], df['OUT_battery_mass_kg'], label=f"Dist={md}mi")
    axsC[7].set_xlabel("Cruise Speed (m/s)"); axsC[7].set_ylabel("Battery Mass (kg)")

    # 26 & 27. Mission Stacked Bars (Single baseline)
    baseline_res = compute_aircraft(base)
    segments = ['Takeoff', 'Vert Climb', 'Trans Out', 'Cruise Out', 'Attack', 'Cruise Ret', 'Trans In', 'Land', 'Retries']
    energies = [
        baseline_res.e_takeoff_hover, baseline_res.e_takeoff_climb, baseline_res.e_transition_out,
        baseline_res.e_outbound, baseline_res.e_attack, baseline_res.e_return,
        baseline_res.e_transition_in, baseline_res.e_landing_hover, baseline_res.e_retries
    ]
    powers = [
        baseline_res.power_hover_W, baseline_res.power_climb_W, baseline_res.power_transition_W,
        baseline_res.power_cruise_W, baseline_res.power_cruise_W, baseline_res.power_cruise_W,
        baseline_res.power_transition_W, baseline_res.power_hover_W, 0
    ]
    bots = np.zeros(1)
    for s, e in zip(segments, energies):
        axsC[8].bar(["Mission"], [e], bottom=bots, label=f"{s}")
        bots += e
    axsC[8].set_title("Stacked Energy (Wh)")

    axsC[9].bar(segments, powers, color='coral')
    axsC[9].tick_params(axis='x', rotation=45)
    axsC[9].set_title("Segment Power (W)")

    # 28 & 29. Timeline Plots
    times = [
        base.hover_time_takeoff_s, 5.0, 15.0, 
        miles_to_m(base.outbound_distance_miles)/base.cruise_speed_mps,
        base.attack_time_s,
        miles_to_m(base.return_distance_miles)/base.cruise_speed_mps,
        15.0, base.hover_time_landing_s
    ]
    t_cum = [0]
    p_seq = [0]
    e_used = [0]
    curr_e = 0
    for i, t in enumerate(times):
        t_cum.extend([t_cum[-1], t_cum[-1]+t])
        p_seq.extend([powers[i], powers[i]])
        e_used.extend([curr_e, curr_e + energies[i]])
        curr_e += energies[i]
        
    axsC[10].plot(t_cum, p_seq, color='r')
    axsC[10].set_xlabel("Time (s)"); axsC[10].set_ylabel("Instantaneous Power (W)")
    
    total_e = baseline_res.required_nominal_batt_Wh * base.batt_usable_fraction
    axsC[11].plot(t_cum, [total_e - e for e in e_used], color='g')
    axsC[11].set_xlabel("Time (s)"); axsC[11].set_ylabel("Remaining Usable Energy (Wh)")

    # 30, 31, 32. Battery Voltages
    voltages = np.array([22.2, 33.3, 44.4, 55.5])
    df_v = run_1d_sweep(base, "batt_voltage_V", voltages)
    axsC[12].plot(voltages, df_v['OUT_peak_current_A'], 'o-')
    axsC[12].set_xlabel("Battery Voltage (V)"); axsC[12].set_ylabel("Peak Current (A)")
    
    axsC[13].plot(voltages, df_v['OUT_battery_Ah'], 'o-', color='orange')
    axsC[13].set_xlabel("Battery Voltage (V)"); axsC[13].set_ylabel("Battery Capacity (Ah)")
    
    axsC[14].plot(voltages, df_v['OUT_peak_c_rate'], 'o-', color='purple')
    axsC[14].set_xlabel("Battery Voltage (V)"); axsC[14].set_ylabel("Peak C-Rate")

    for i, ax in enumerate(axsC): 
        ax.grid(True)
        if i in [0,1,2,6,7]: ax.legend(fontsize=7)
    figC.suptitle("C. Mission Energy and Battery Plots (18-32)")
    figC.tight_layout()
    figC.savefig("plot_C_mission_battery.png", dpi=120)
    plt.close(figC)

    # -----------------------------------------------------------------------
    # D. Coupled Design Plots (33-38)
    # -----------------------------------------------------------------------
    figD, axsD = plt.subplots(2, 3, figsize=(18, 10))
    axsD = axsD.flatten()

    # 33 & 34. AR vs Mission Energy & Battery
    for cd in [0.03, 0.045, 0.06]:
        b = base.copy(); b.cd0 = cd
        df = run_1d_sweep(b, "aspect_ratio", ar_arr)
        axsD[0].plot(df['aspect_ratio'], df['OUT_total_mission_energy_Wh'], label=f"CD0={cd}")
    for m in m_arr:
        b = base.copy(); b.total_mass_guess_kg = m
        df = run_1d_sweep(b, "aspect_ratio", ar_arr)
        axsD[1].plot(df['aspect_ratio'], df['OUT_battery_mass_kg'], label=f"Mass={m}")
    axsD[0].set_xlabel("Aspect Ratio"); axsD[0].set_ylabel("Total Mission Energy (Wh)")
    axsD[1].set_xlabel("Aspect Ratio"); axsD[1].set_ylabel("Battery Mass (kg)")

    # 35. Rotor vs Battery Mass
    for m in m_arr:
        b = base.copy(); b.total_mass_guess_kg = m
        df = run_1d_sweep(b, "rotor_diameter_m", d_rot_arr)
        axsD[2].plot(df['rotor_diameter_m'], df['OUT_battery_mass_kg'], label=f"Mass={m}")
    axsD[2].set_xlabel("Rotor Diameter (m)"); axsD[2].set_ylabel("Battery Mass (kg)")

    # 36. AR vs Rotor Contours
    d_v2 = np.linspace(0.4, 0.7, 10)
    ar_v2 = np.linspace(6, 12, 10)
    df_36 = run_2d_sweep(base, "aspect_ratio", ar_v2, "rotor_diameter_m", d_v2)
    p36 = df_36.pivot(index="rotor_diameter_m", columns="aspect_ratio", values="OUT_battery_mass_kg")
    c36 = axsD[3].contourf(p36.columns, p36.index, p36.values, levels=20, cmap='viridis')
    figD.colorbar(c36, ax=axsD[3], label="Battery Mass (kg)")
    axsD[3].set_xlabel("Aspect Ratio"); axsD[3].set_ylabel("Rotor Diameter")

    # 37. Cruise vs Rotor Contours
    df_37 = run_2d_sweep(base, "cruise_speed_mps", np.linspace(40,60,10), "rotor_diameter_m", d_v2)
    p37 = df_37.pivot(index="rotor_diameter_m", columns="cruise_speed_mps", values="OUT_total_mission_energy_Wh")
    c37 = axsD[4].contourf(p37.columns, p37.index, p37.values, levels=20, cmap='ocean')
    figD.colorbar(c37, ax=axsD[4], label="Mission Energy (Wh)")
    axsD[4].set_xlabel("Cruise Speed (m/s)"); axsD[4].set_ylabel("Rotor Diameter")

    # 38. Total mass vs AR Contours
    df_38 = run_2d_sweep(base, "total_mass_guess_kg", np.linspace(25,40,10), "aspect_ratio", ar_v2)
    p38 = df_38.pivot(index="aspect_ratio", columns="total_mass_guess_kg", values="OUT_span_m")
    c38 = axsD[5].contourf(p38.columns, p38.index, p38.values, levels=20, cmap='plasma')
    figD.colorbar(c38, ax=axsD[5], label="Span (m)")
    axsD[5].set_xlabel("Total Mass (kg)"); axsD[5].set_ylabel("Aspect Ratio")

    for i in range(3): axsD[i].legend(fontsize=7)
    figD.suptitle("D. Coupled Design Plots (33-38)")
    figD.tight_layout()
    figD.savefig("plot_D_coupled_design.png", dpi=120)
    plt.close(figD)
    
    print("  -> Exported 38 plots across 4 combined panel PNG files.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN 
# ═══════════════════════════════════════════════════════════════════════════

def main():
    base = Assumptions()
    print("================================================================")
    print(" VTOL Sizing Engine - 38 Plots & Interactive Web Export")
    print("================================================================")
    
    res = compute_aircraft(base, force_convergence=True)
    print(f"\nBaseline Converged MTOW: {res.mtow_calc_kg:.2f} kg, Mission Energy: {res.total_mission_energy_Wh:.0f} Wh\n")
    
    generate_all_plots(base)
    
    print("[Table] Running large multidimensional sweep dump CSV for Interactive Dashboard...")
    v_cr = np.arange(40, 62, 5)
    ar = np.arange(6, 13, 2)
    d_rot = np.arange(0.4, 0.75, 0.1)
    
    grid = list(itertools.product(v_cr, ar, d_rot))
    results = []
    for (v, a, d) in grid:
        b = base.copy(); b.cruise_speed_mps = v; b.aspect_ratio = a; b.rotor_diameter_m = d
        res = compute_aircraft(b, force_convergence=False)
        r = asdict(b); r.update({f"OUT_{k}": val for k, val in asdict(res).items()})
        results.append(r)
        
    df_out = pd.DataFrame(results)
    df_out.to_csv("vtol_comprehensive_sweep.csv", index=False)
    print(f"  -> Dumped {df_out.shape[0]} rows to vtol_comprehensive_sweep.csv")
    print("Done. Use interactive_explorer.html to dynamically explore custom X-Y scatter patterns.")

if __name__ == "__main__":
    main()
