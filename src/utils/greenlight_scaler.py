from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


class GreenlightScaler:
    def __init__(self):
        self.scaling_ranges = {
            "iGlob": (0, 1200),
            "tOut": (-20, 40),
            "vpOut": (0, 7300),
            "co2Out": (
                0,
                1000,
            ),  # @gsoykan - this seems problematic because values hover around 0
            "wind": (0, 45),
            "tSky": (-50, 30),
            "tSoOut": (-5, 35),
            **self.output_scaling_ranges,  # outdoor
            "shScr": (0, 1),
            "blScr": (0, 1),
            "roof": (0, 1),
            "tPipe": (-10, 100),
            "tGroPipe": (-10, 100),
            "lamp": (0, 1),
            "intLamp": (0, 1),
            "extCo2": (0, 1),
            "sideLee": (0, 1),
            "sideWind": (0, 1),  # controls
            "tCanSum": (0, 3500),  # °C day
            "cBuf": (0, 30000),  # mg{CH2O} m⁻²
            "cLeaf": (0, 200000),  # mg{CH2O} m⁻²
            "cStem": (0, 300000),  # mg{CH2O} m⁻²
            "cFruit": (0, 300000),  # mg{CH2O} m⁻²
            "lai": (
                0,
                6,
            ),  # m² {leaf} m⁻² {floor} # crop (values approximated from the dataset)
            # AUX States...
            "qRail": (0, 500),  # W m⁻²
            "qGro": (0, 500),  # W m⁻²
            "lampIn": (0, 1000),  # W m⁻²
            "intLampIn": (0, 1000),  # W m⁻²
            "lampCool": (0, 200),  # W m⁻²
            "vent": (0, 1),  # m³ m⁻² s⁻¹
            "co2Inj": (0, 50),  # mg m⁻² s⁻¹
            # Crop model auxiliary states
            "mcAirBuf": (0, 1),  # mg{CH₂O} m⁻² s⁻¹
            "mcBufLeaf": (0, 1),  # mg{CH₂O} m⁻²
            "mcBufFruit": (0, 1),  # mg{CH₂O} m⁻²
            "mcBufStem": (0, 1),  # mg{CH₂O} m⁻²
            "mcBufAir": (0, 1),  # mg{CH₂O} m⁻²
            "mcLeafAir": (0, 1),  # mg{CH₂O} m⁻²
            "mcFruitAir": (0, 20),  # mg{CH₂O} m⁻²
            "mcStemAir": (0, 1),  # mg{CH₂O} m⁻²
            "mcLeafHar": (0, 5),  # mg{CH₂O} m⁻²
            "mcFruitHar": (0, 20),  # mg{CH₂O} m⁻²
            "mcAirCan": (0, 5),  # mg{CH₂O} m⁻²
            "mvCanAir": (0, 0.01),  # kg m⁻² s⁻¹
        }

    @property
    def output_scaling_ranges(self):
        return {
            "tAir": (-20, 40),
            "vpAir": (0, 7300),
            "co2Air": (0, 5000),  # output
        }

    @property
    def parameter_scaling_ranges(self):
        # in the new world params 25 of them are changing
        # remaining 230 are constant
        return {
            # 1
            "alfaLeafAir": (0, 10),  # Convective heat exchange coefficient
            "L": (2.4e6, 2.5e6),  # Latent heat of evaporation
            "sigma": (5.6e-8, 5.7e-8),  # Stefan-Boltzmann constant
            "epsCan": (0, 1),  # FIR emission coefficient of canopy
            "epsSky": (0, 1),  # FIR emission coefficient of sky
            "etaGlobNir": (0, 1),  # Ratio of NIR in global radiation
            "etaGlobPar": (0, 1),  # Ratio of PAR in global radiation
            # 2
            "etaMgPpm": (0.5, 0.6),  # CO2 conversion factor
            "etaRoofThr": (0, 1),  # Ratio between roof vent area and total vent area
            "rhoAir0": (1, 1.5),  # Density of air at sea level
            "rhoCanPar": (0, 0.1),  # PAR reflection coefficient
            "rhoCanNir": (0.3, 0.4),  # NIR reflection coefficient
            "rhoSteel": (7800, 7900),  # Density of steel
            "rhoWater": (999, 1001),  # Density of water
            "gamma": (60, 70),  # Psychrometric constant
            "omega": (1.9e-7, 2e-7),  # Yearly frequency to calculate soil temperature
            "capLeaf": (1e3, 1.3e3),  # Heat capacity of canopy leaves
            "cEvap1": (4, 5),  # Radiation effect coefficient
            "cEvap2": (0.5, 0.6),  # Radiation effect coefficient
            # 3
            "cEvap3Day": (5.0e-7, 7.0e-7),  # Coefficient for CO2 effect (day)
            "cEvap3Night": (1.0e-11, 1.2e-11),  # Coefficient for CO2 effect (night)
            "cEvap4Day": (
                4.0e-6,
                4.6e-6,
            ),  # Coefficient for vapor pressure effect (day)
            "cEvap4Night": (
                5.0e-6,
                5.4e-6,
            ),  # Coefficient for vapor pressure effect (night)
            "cPAir": (900, 1100),  # Specific heat capacity of air
            "cPSteel": (600, 700),  # Specific heat capacity of steel
            "cPWater": (4.1e3, 4.2e3),  # Specific heat capacity of water
            "g": (9.8, 9.82),  # Acceleration of gravity
            "hSo1": (0.03, 0.05),  # Thickness of soil layer 1
            "hSo2": (0.07, 0.09),  # Thickness of soil layer 2
            "hSo3": (0.15, 0.17),  # Thickness of soil layer 3
            "hSo4": (0.31, 0.33),  # Thickness of soil layer 4
            "hSo5": (0.63, 0.65),  # Thickness of soil layer 5
            "k1Par": (0.6, 0.8),  # PAR extinction coefficient
            "k2Par": (
                0.6,
                0.8,
            ),  # PAR extinction coefficient for light reflected from the floor
            "kNir": (0.25, 0.3),  # NIR extinction coefficient
            "kFir": (0.9, 1.0),  # FIR extinction coefficient
            "mAir": (28.9, 29.1),  # Molar mass of air
            "hSoOut": (1.2, 1.3),  # Thickness of the external soil layer
            # 4
            "mWater": (17, 19),  # Molar mass of water (kg/kmol)
            "R": (8200, 8400),  # Molar gas constant (J/kmol·K)
            "rCanSp": (0, 10),  # Radiation value above the canopy (W/m^2)
            "rB": (200, 350),  # Boundary layer resistance (s/m)
            "rSMin": (50, 100),  # Minimum canopy resistance (s/m)
            "sRs": (-2, 0),  # Slope of stomatal resistance model (m/W^2)
            "etaGlobAir": (0, 0.2),  # Ratio of global radiation absorbed
            "psi": (0, 60),  # Greenhouse cover slope (degrees)
            "aFlr": (1, 1e5),  # Floor area of greenhouse (m^2)
            "aCov": (1, 1e6),  # Cover surface area (m^2)
            "hAir": (1, 10.0),  # Height of main compartment (m)
            "hGh": (1.0, 10),  # Mean height of greenhouse (m)
            # cHecIn => high was 2.0 now 10.0
            "cHecIn": (1.5, 10.0),  # Heat exchange coefficient (W/m^2·K)
            "cHecOut1": (2.5, 3.0),  # Heat exchange parameter (W/m^2·K)
            "cHecOut2": (1.0, 1.5),  # Heat exchange parameter (J/m^3·K)
            "cHecOut3": (0.5, 1.5),  # Heat exchange parameter (-)
            "hElevation": (-10, 2000),  # Altitude (m above sea level)
            "aRoof": (1, 1e5),  # Roof ventilation area (m^2)
            "hVent": (0.5, 2.0),  # Ventilation opening height (m)
            "etaInsScr": (0, 1),  # Porosity of insect screen (-)
            "aSide": (0, 100),  # Side ventilation area (m^2)
            "cDgh": (0.5, 1),  # Ventilation discharge coefficient (-)
            "cLeakage": (1e-5, 5e-4),  # Leakage coefficient (-)
            "cWgh": (0.05, 0.1),  # Wind pressure coefficient (-)
            "hSideRoof": (
                0,
                1,
            ),  # Vertical distance between side and roof ventilation (m)
            # 5
            "epsRfFir": (0.8, 1.0),  # FIR emission coefficient of the roof
            "rhoRf": (900, 2.7e3),  # Density of the roof layer (kg/m^3)
            "rhoRfNir": (0.1, 0.2),  # NIR reflection coefficient of the roof
            "rhoRfPar": (0.1, 0.2),  # PAR reflection coefficient of the roof
            "rhoRfFir": (0.1, 0.2),  # FIR reflection coefficient of the roof
            # bottom was 0.8 now 0 (for both tauRfNir & tauRfPar)
            "tauRfNir": (0.0, 0.9),  # NIR transmission coefficient of the roof
            "tauRfPar": (0.0, 0.9),  # PAR transmission coefficient of the roof
            "tauRfFir": (0, 0.1),  # FIR transmission coefficient of the roof
            "lambdaRf": (0.02, 1.4),  # Thermal heat conductivity of the roof (W/m·K)
            "cPRf": (0.75e3, 2.3e3),  # Specific heat capacity of roof layer (J/K·kg)
            "hRf": (1e-3, 1e-2),  # Thickness of roof layer (m)
            # 6
            "epsShScrPerFir": (
                0,
                1,
            ),  # FIR emission coefficient of the whitewash (no whitewash)
            "rhoShScrPer": (0, 1),  # Density of the whitewash (no whitewash)
            "rhoShScrPerNir": (
                0,
                1,
            ),  # NIR reflection coefficient of the whitewash (no whitewash)
            "rhoShScrPerPar": (
                0,
                1,
            ),  # PAR reflection coefficient of the whitewash (no whitewash)
            "rhoShScrPerFir": (
                0,
                1,
            ),  # FIR reflection coefficient of the whitewash (no whitewash)
            "tauShScrPerNir": (
                0,
                1,
            ),  # NIR transmission coefficient of the whitewash (no whitewash)
            "tauShScrPerPar": (
                0,
                1,
            ),  # PAR transmission coefficient of the whitewash (no whitewash)
            "tauShScrPerFir": (
                0,
                1,
            ),  # FIR transmission coefficient of the whitewash (no whitewash)
            "cPShScrPer": (
                0,
                1,
            ),  # Specific heat capacity of the whitewash (no whitewash)
            "hShScrPer": (0, 1),  # Thickness of the whitewash (no whitewash)
            # 7
            "rhoShScrNir": (
                0,
                1,
            ),  # NIR reflection coefficient of the shadow screen (no shadow screen)
            "rhoShScrPar": (
                0,
                1,
            ),  # PAR reflection coefficient of the shadow screen (no shadow screen)
            "rhoShScrFir": (
                0,
                1,
            ),  # FIR reflection coefficient of the shadow screen (no shadow screen)
            "tauShScrNir": (
                0,
                1,
            ),  # NIR transmission coefficient of the shadow screen (no shadow screen)
            "tauShScrPar": (
                0,
                1,
            ),  # PAR transmission coefficient of the shadow screen (no shadow screen)
            "tauShScrFir": (
                0,
                1,
            ),  # FIR transmission coefficient of the shadow screen (no shadow screen)
            "etaShScrCd": (
                0,
                1,
            ),  # Effect of shadow screen on discharge coefficient (no shadow screen)
            "etaShScrCw": (
                0,
                1,
            ),  # Effect of shadow screen on wind pressure coefficient (no shadow screen)
            "kShScr": (0, 1),  # Shadow screen flux coefficient (no shadow screen)
            # 8
            "epsThScrFir": (
                0.6,
                0.95,
            ),  # FIR emission coefficient of the thermal screen
            "rhoThScr": (20, 500),  # Density of thermal screen (kg/m^3)
            "rhoThScrNir": (
                0.1,
                0.95,
            ),  # NIR reflection coefficient of the thermal screen
            "rhoThScrPar": (
                0.05,
                0.85,
            ),  # PAR reflection coefficient of the thermal screen
            "rhoThScrFir": (
                0.1,
                0.2,
            ),  # FIR reflection coefficient of the thermal screen
            "tauThScrNir": (
                0.1,
                0.9,
            ),  # NIR transmission coefficient of the thermal screen
            "tauThScrPar": (
                0.05,
                0.85,
            ),  # PAR transmission coefficient of the thermal screen
            "tauThScrFir": (
                0.1,
                0.2,
            ),  # FIR transmission coefficient of the thermal screen
            "cPThScr": (
                1e2,
                2.5e3,
            ),  # Specific heat capacity of the thermal screen (J/kg·K)
            "hThScr": (1e-4, 1e-3),  # Thickness of the thermal screen (m)
            "kThScr": (
                1e-5,
                1e-3,
            ),  # Thermal screen flux coefficient (m^3/m^2·K^(-2/3)·s^(-1))
            # 9
            "epsBlScrFir": (0, 1),  # FIR emission coefficient of the blackout screen
            "rhoBlScr": (100, 500),  # Density of blackout screen (kg/m^3)
            "rhoBlScrNir": (0, 1),  # NIR reflection coefficient of blackout screen
            "rhoBlScrPar": (0, 1),  # PAR reflection coefficient of blackout screen
            "tauBlScrNir": (
                0.0,
                0.1,
            ),  # NIR transmission coefficient of blackout screen
            "tauBlScrPar": (
                0.0,
                0.1,
            ),  # PAR transmission coefficient of blackout screen
            "tauBlScrFir": (0, 1),  # FIR transmission coefficient of blackout screen
            "cPBlScr": (
                1e3,
                2.5e3,
            ),  # Specific heat capacity of blackout screen (J/kg·K)
            "hBlScr": (1e-4, 1e-3),  # Thickness of blackout screen (m)
            "kBlScr": (
                1e-5,
                # it was 1e-4 - now 1e-3
                1e-3,
            ),  # Blackout screen flux coefficient (m^3/m^2·K^(-2/3)·s^(-1))
            # 10
            "epsFlr": (0, 2.0),  # FIR emission coefficient of the floor
            "rhoFlr": (1e3, 2.5e3),  # Density of the floor (kg/m^3)
            "rhoFlrNir": (0, 1),  # NIR reflection coefficient of the floor
            "rhoFlrPar": (0, 1),  # PAR reflection coefficient of the floor
            "lambdaFlr": (1, 2.5),  # Thermal heat conductivity of the floor (W/m·K)
            "cPFlr": (1e2, 1e3),  # Specific heat capacity of the floor (J/kg·K)
            "hFlr": (0.01, 0.05),  # Thickness of floor (m)
            # 11
            "rhoCpSo": (1e6, 2e6),  # Volumetric heat capacity of the soil (J/m^3·K)
            "lambdaSo": (0, 1),  # Thermal heat conductivity of the soil layers (W/m·K)
            "epsPipe": (0, 1),  # FIR emission coefficient of the heating pipes
            "phiPipeE": (1e-2, 1e-1),  # External diameter of pipes (m)
            "phiPipeI": (1e-3, 1e-1),  # Internal diameter of pipes (m)
            "lPipe": (
                0,
                2.5,
            ),  # Length of heating pipes per greenhouse floor area (m/m^2)
            # 12000000.0
            "pBoil": (
                1e6,
                15e6,
            ),  # Capacity of the heating system (W), scaled with greenhouse floor area
            # 12
            "phiExtCo2": (5e5, 1e6),  # Capacity of external CO2 source (mg/s)
            "capPipe": (1e3, 2e4),  # Heat capacity of heating pipes (J/K·m^2)
            "rhoAir": (1.0, 1.5),  # Density of the air (kg/m^3)
            "capAir": (1e3, 1e4),  # Heat capacity of air in main compartment (J/K·m^2)
            "capFlr": (0, 1e5),  # Heat capacity of the floor (J/K·m^2)
            "capSo1": (0, 1e5),  # Heat capacity of soil layer 1 (J/K·m^2)
            "capSo2": (0, 5e5),  # Heat capacity of soil layer 2 (J/K·m^2)
            "capSo3": (0, 5e5),  # Heat capacity of soil layer 3 (J/K·m^2)
            "capSo4": (0, 1e6),  # Heat capacity of soil layer 4 (J/K·m^2)
            "capSo5": (0, 1e6),  # Heat capacity of soil layer 5 (J/K·m^2)
            "capThScr": (0, 1000),  # Heat capacity of thermal screen (J/K·m^2)
            "capTop": (0, 1000),  # Heat capacity of air in top compartments (J/K·m^2)
            "capBlScr": (0, 1000),  # Heat capacity of blackout screen (J/K·m^2)
            "capCo2Air": (0, 10),  # Capacity for CO2 in main compartment (m)
            "capCo2Top": (0, 1),  # Capacity for CO2 in top compartment (m)
            "aPipe": (0, 0.5),  # Surface of pipes for floor area (-)
            "fCanFlr": (0, 1),  # View factor from canopy to floor (-)
            "pressure": (
                50000,
                150000,
            ),  # Absolute air pressure at given elevation (Pa)
            # 13
            "globJtoUmol": (2.0, 2.5),  # Conversion factor from global radiation to PAR
            "j25LeafMax": (
                200,
                220,
            ),  # Maximal rate of electron transport at 25°C (umol e-/m^2·s)
            "cGamma": (
                1.5,
                2.0,
            ),  # Effect of canopy temperature on CO2 compensation point (umol CO2/mol air·K)
            "etaCo2AirStom": (
                0.6,
                0.7,
            ),  # Conversion from greenhouse air CO2 to stomatal CO2
            "eJ": (36e3, 38e3),  # Activation energy for Jpot calculation (J/mol)
            "t25k": (297, 299),  # Reference temperature for Jpot calculation (K)
            "S": (700, 720),  # Entropy term for Jpot calculation (J/mol·K)
            "H": (21e4, 23e4),  # Deactivation energy for Jpot calculation (J/mol)
            "theta": (
                0.6,
                0.8,
            ),  # Degree of curvature of the electron transport rate (-)
            "alpha": (0.35, 0.4),  # Conversion factor from photons to electrons (-)
            "mCh2o": (29e-3, 31e-3),  # Molar mass of CH2O (mg/umol)
            "mCo2": (43e-3, 45e-3),  # Molar mass of CO2 (mg/umol)
            "parJtoUmolSun": (
                4.5,
                4.7,
            ),  # Conversion factor of sun's PAR from J (umol photons/J)
            "laiMax": (2.5, 3.5),  # Leaf area index (m^2/m^2)
            "sla": (2.5e-5, 2.8e-5),  # Specific leaf area (m^2/mg)
            "rgr": (2.5e-6, 3.5e-6),  # Relative growth rate (kg/kg·s)
            # 14
            "cFruitMax": (2e5, 4e5),  # Maximum fruit size (mg/m^2)
            "cFruitG": (0.1, 0.5),  # Fruit growth respiration coefficient
            "cLeafG": (0.1, 0.5),  # Leaf growth respiration coefficient
            "cStemG": (0.1, 0.5),  # Stem growth respiration coefficient
            "cRgr": (
                2e6,
                3e6,
            ),  # Regression coefficient in maintenance respiration function (s^-1)
            "q10m": (
                1,
                3,
            ),  # Q10 value of temperature effect on maintenance respiration (-)
            "cFruitM": (
                0.5e-7,
                2e-7,
            ),  # Fruit maintenance respiration coefficient (mg/mg·s)
            "cLeafM": (
                1e-7,
                5e-7,
            ),  # Leaf maintenance respiration coefficient (mg/mg·s)
            "cStemM": (
                0.5e-7,
                2e-7,
            ),  # Stem maintenance respiration coefficient (mg/mg·s)
            "rgFruit": (0.1, 0.5),  # Potential fruit growth coefficient (mg/m^2·s)
            "rgLeaf": (0.05, 0.15),  # Potential leaf growth coefficient (mg/m^2·s)
            "rgStem": (0.05, 0.15),  # Potential stem growth coefficient (mg/m^2·s)
            # 15
            "cBufMax": (15e3, 25e3),  # Maximum capacity of carbohydrate buffer (mg/m^2)
            "cBufMin": (500, 1500),  # Minimum capacity of carbohydrate buffer (mg/m^2)
            "tCan24Max": (
                10,
                40,
            ),  # Inhibition of carbohydrate flow due to high daily average temperatures (°C)
            "tCan24Min": (
                5,
                25,
            ),  # Inhibition of carbohydrate flow due to low daily average temperatures (°C)
            "tCanMax": (
                10,
                50,
            ),  # Inhibition of carbohydrate flow due to high instantaneous temperatures (°C)
            "tCanMin": (
                5,
                20,
            ),  # Inhibition of carbohydrate flow due to low instantaneous temperatures (°C)
            "tEndSum": (
                500,
                2000,
            ),  # Temperature sum where crop is fully generative (°C day)
            "rhMax": (0, 100),  # Upper bound on relative humidity (%)
            "dayThresh": (
                10,
                30,
            ),  # Threshold for day-night switch based on radiation (W/m^2)
            "tSpDay": (0, 40),  # Heating setpoint during the day (°C)
            "tSpNight": (0, 40),  # Heating setpoint during the night (°C)
            "tHeatBand": (-10, 10),  # P-band for heating (°C)
            "tVentOff": (
                0,
                2,
            ),  # Distance from heating setpoint where ventilation stops (°C)
            "tScreenOn": (
                0,
                5,
            ),  # Distance from screen setpoint where screen is activated (°C)
            "thScrSpDay": (0, 10),  # Screen activation temperature during the day (°C)
            "thScrSpNight": (
                0,
                20,
            ),  # Screen activation temperature during the night (°C)
            "thScrPband": (-10, 10),  # P-band for thermal screen (°C)
            "co2SpDay": (0, 1500),  # CO2 supply setpoint during the day (ppm)
            "co2Band": (-250, 0),  # P-band for CO2 supply (ppm)
            "heatDeadZone": (
                0,
                10,
            ),  # Dead zone between heating and ventilation setpoints (°C)
            "ventHeatPband": (0, 10),  # P-band for ventilation due to excess heat (°C)
            "ventColdPband": (
                -10,
                10,
            ),  # P-band for ventilation due to low temperature (°C)
            "ventRhPband": (
                0,
                100,
            ),  # P-band for ventilation due to relative humidity (%)
            "thScrRh": (
                -10,
                10,
            ),  # Relative humidity threshold for thermal screen opening (% deviation from rhMax)
            "thScrRhPband": (
                0,
                10,
            ),  # P-band for thermal screen opening due to excess relative humidity (%)
            "thScrDeadZone": (
                0,
                10,
            ),  # Dead zone between heating and thermal screen opening (°C)
            "lampsOn": (
                -1,
                25,
            ),  # Time of day to switch on lamps (hours since midnight)
            "lampsOff": (
                -1,
                25,
            ),  # Time of day to switch off lamps (hours since midnight)
            # Lamp control properties
            "dayLampStart": (
                -2,
                365,
            ),  # Day of the year when lamps start (-1: no influence)
            "dayLampStop": (
                0,
                365,
            ),  # Day of the year when lamps stop (> 366: no influence)
            "lampsOffSun": (
                0,
                1000,
            ),  # Global radiation threshold above which lamps are switched off (W/m^2)
            "lampRadSumLimit": (
                0,
                20,
            ),  # Predicted daily radiation sum where lamps are not used (MJ/m^2/day)
            "lampExtraHeat": (0, 5),  # Additional heat limit for lamps control (°C)
            # Blackout screen properties
            "blScrExtraRh": (
                0,
                100,
            ),  # Relative humidity threshold for blackout screen control (%)
            "useBlScr": (0, 1),  # Blackout screen usage (0: not used, 1: used)
            # Mechanical cooling and dehumidification
            "mechCoolPband": (0, 5),  # P-band for mechanical cooling (°C)
            "mechDehumidPband": (0, 5),  # P-band for mechanical dehumidification (%)
            "heatBufPband": (-5, 0),  # P-band for heating from buffer (°C)
            "mechCoolDeadZone": (
                0,
                5,
            ),  # Dead zone between heating and mechanical cooling setpoints (°C)
            # 16
            "epsGroPipe": (0, 1),  # Emissivity of grow pipes [-]
            "lGroPipe": (
                0,
                2,
            ),  # Length of grow pipes per greenhouse floor area (m/m^2)
            "phiGroPipeE": (1e-2, 5e-2),  # External diameter of grow pipes (m)
            "phiGroPipeI": (1e-2, 5e-2),  # Internal diameter of grow pipes (m)
            "aGroPipe": (
                0.05,
                0.5,
            ),  # Surface area of grow pipes for floor area (m^2/m^2)
            "pBoilGro": (0, 10000),  # Capacity of the grow pipe heating system (W)
            # Heat capacity of grow pipes [J K^-1 m^-2]
            "capGroPipe": (
                0,
                1e4,
            ),  # Heat capacity of grow pipes (adjustable based on real values)
            "thetaLampMax": (0, 300),  # Maximum intensity of lamps (W/m^2)
            "heatCorrection": (
                0,
                2,
            ),  # Temperature setpoint correction when lamps are on (°C)
            "etaLampPar": (0, 1),  # Fraction of lamp input converted to PAR [-]
            "etaLampNir": (0, 1),  # Fraction of lamp input converted to NIR [-]
            "tauLampPar": (0, 1.0),  # Transmissivity of lamp layer to PAR [-]
            "rhoLampPar": (0, 1),  # Reflectivity of lamp layer to PAR [-]
            "tauLampNir": (0, 1.0),  # Transmissivity of lamp layer to NIR [-]
            "rhoLampNir": (0, 1.0),  # Reflectivity of lamp layer to NIR [-]
            "tauLampFir": (0, 1.0),  # Transmissivity of lamp layer to FIR [-]
            "aLamp": (0, 0.1),  # Lamp area (m^2/m^2 floor)
            "epsLampTop": (0, 1),  # Emissivity of top side of lamp [-]
            "epsLampBottom": (0, 1),  # Emissivity of bottom side of lamp [-]
            "capLamp": (0, 400),  # Heat capacity of lamp (J/K/m^2)
            "cHecLampAir": (0, 5),  # Heat exchange coefficient of lamp (W/m^2/K)
            "etaLampCool": (0, 1),  # Fraction of lamp input removed by cooling [-]
            "zetaLampPar": (0, 10),  # J to umol conversion of PAR output of lamp
            # 17
            "vIntLampPos": (
                0,
                1,
            ),  # Vertical position of the interlights within the canopy [-]
            "fIntLampDown": (
                0,
                1,
            ),  # Fraction of interlight light output that goes downwards [-]
            "capIntLamp": (0, 20),  # Heat capacity of interlight lamps (J/K/m^2)
            "etaIntLampPar": (
                0,
                1,
            ),  # Fraction of interlight lamp input converted to PAR [-]
            "etaIntLampNir": (
                0,
                1,
            ),  # Fraction of interlight lamp input converted to NIR [-]
            "aIntLamp": (0, 0.1),  # Interlight lamp area (m^2/m^2 floor)
            "epsIntLamp": (0, 1),  # Emissivity of interlight [-]
            "thetaIntLampMax": (0, 1),  # Maximum intensity of interlight lamps (W/m^2)
            "zetaIntLampPar": (0, 0.1),
            # Conversion from Joules to umol photons within the PAR output of the interlight
            "cHecIntLampAir": (
                0,
                1,
            ),  # Heat exchange coefficient of interlights (W/m^2/K)
            "tauIntLampFir": (
                0,
                1.0,
            ),  # Transmissivity of FIR through the interlights [-]
            "k1IntPar": (
                0,
                2,
            ),  # PAR extinction coefficient of the canopy for light from interlights [-]
            "k2IntPar": (0, 2),
            # PAR extinction coefficient of the canopy for light from interlights through the floor [-]
            "kIntNir": (
                0,
                2,
            ),  # NIR extinction coefficient of the canopy for light from interlights [-]
            "kIntFir": (
                0,
                2.0,
            ),  # FIR extinction coefficient of the canopy for light from interlights [-]
            # Other parameters
            "cLeakTop": (
                0,
                1,
            ),  # Fraction of leakage ventilation going from the top [-]
            "minWind": (
                0,
                1,
            ),  # Wind speed where the effect of wind on leakage begins (m/s)
            #
            "cLeafMax": (25000, 250000),
            # "lambdaShScrPer": (nan, nan),
        }

    @staticmethod
    def transform_json_dict(
        json_dict: Dict[str, int | float],
        scaling_ranges: Dict[str, Tuple[int | float, int | float]],
        enforce_rescaling_all_json_keys: bool = False,
        enforce_rescaling_all_scale_ranges: bool = False,
    ) -> Dict[str, int | float]:
        result = {}

        for key, (min_val, max_val) in scaling_ranges.items():
            if key in json_dict:
                result[key] = (json_dict[key] - min_val) / (max_val - min_val)
            elif enforce_rescaling_all_scale_ranges:
                raise AssertionError(
                    f"enforce_rescaling_all_scale_ranges is enabled, {key} is not in json_dict"
                )

        if enforce_rescaling_all_json_keys:
            json_keys = set(json_dict.keys())
            result_keys = set(result.keys())
            assert (
                json_keys == result_keys
            ), f"enforce_rescaling_all_json_keys is enabled, {json_keys.difference(result_keys)} are not in result"

        return result

    def transform(
        self,
        dataset: pd.DataFrame | np.ndarray,
        is_only_output: bool = False,
    ) -> pd.DataFrame:
        if is_only_output:
            scaling_ranges = self.output_scaling_ranges
        else:
            scaling_ranges = self.scaling_ranges

        if isinstance(dataset, pd.DataFrame):
            for column, (min_val, max_val) in scaling_ranges.items():
                if column in dataset.columns:
                    dataset[column] = (dataset[column] - min_val) / (max_val - min_val)
        # TODO: @gsoykan - make sure indices work as expected...
        elif isinstance(dataset, np.ndarray):
            for i, (column, (min_val, max_val)) in enumerate(scaling_ranges.items()):
                dataset[:, i] = (dataset[:, i] - min_val) / (max_val - min_val)
        return dataset

    def inverse_transform(
        self,
        dataset: pd.DataFrame | np.ndarray | torch.Tensor,
        is_only_output: bool = False,
    ) -> pd.DataFrame:
        if is_only_output:
            scaling_ranges = self.output_scaling_ranges
        else:
            scaling_ranges = self.scaling_ranges

        if isinstance(dataset, pd.DataFrame):
            for column, (min_val, max_val) in scaling_ranges.items():
                if column in dataset.columns:
                    dataset[column] = dataset[column] * (max_val - min_val) + min_val
        elif isinstance(dataset, np.ndarray) or isinstance(dataset, torch.Tensor):
            for i, (column, (min_val, max_val)) in enumerate(scaling_ranges.items()):
                dataset[:, i] = dataset[:, i] * (max_val - min_val) + min_val

        return dataset
