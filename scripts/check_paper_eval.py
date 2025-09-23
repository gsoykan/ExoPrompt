from typing import Union
import numpy as np
import pandas as pd


def sat_vp(temp):
    """
    Calculate the saturated vapor pressure (Pa) at a given temperature (°C).

    Parameters:
    temp (float): Temperature in Celsius.

    Returns:
    float: Saturated vapor pressure in Pascals.
    """

    # Parameters used in the conversion
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]

    # Saturation vapor pressure calculation
    sat = p[0] * np.exp(p[2] * temp / (temp + p[1]))
    return sat


# Inferred from GreenLight, and below two functions which are implemented in GreenLight Plus
def vapor_rh2pres(temp, rh):
    """
    Converts temperature and relative humidity to vapor pressure.

    Args:
        temp (float or np.ndarray): Temperature in degrees Celsius
        rh (float or np.ndarray): Relative humidity as a decimal (0 to 1)

    Returns:
        float or np.ndarray: Vapor pressure in Pascals (Pa)
    """
    # Conversion parameters
    p = np.array([610.78, 238.3, 17.2694, -6140.4, 273, 28.916])
    sat_p = p[0] * np.exp(p[2] * temp / (temp + p[1]))  # Saturation vapor pressure
    return sat_p * rh


def vapor_dens2pres(
    temp: Union[float, np.ndarray], vapor_dens: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Converts vapor density to vapor pressure for given temperature and vapor density values.

    Args:
        temp (float or np.ndarray): Temperatures in degrees Celsius
        vapor_dens (float or np.ndarray): Vapor density in kg{H2O} m^{-3}

    Returns:
        float or np.ndarray: Vapor pressure in Pascals (Pa)
    """
    p = np.array([610.78, 238.3, 17.2694, -6140.4, 273, 28.916])
    rh = vapor_dens / rh2vapor_dens(temp, 100)  # Relative humidity (0-1)
    sat_p = p[0] * np.exp(p[2] * temp / (temp + p[1]))
    return sat_p * rh


def rh2vapor_dens(
    temp: Union[float, np.ndarray], rh: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Converts relative humidity to vapor density.

    Args:
        temp (float or np.ndarray): Temperatures in degrees Celsius
        rh (float or np.ndarray): Relative humidity as a percentage (0-100)

    Returns:
        float or np.ndarray: Vapor density in kg{H2O} m^{-3}
    """
    R, C2K, Mw = 8.3144598, 273.15, 18.01528e-3
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    sat_p = p[0] * np.exp(p[2] * temp / (temp + p[1]))
    pascals = (rh / 100) * sat_p
    return pascals * Mw / (R * (temp + C2K))


def calculate_rrmse(gt: np.array, pred: np.array) -> float:
    """
    Calculates Relative Root Mean Square Error (RRMSE).

    Args:
        gt (np.array): Ground truth values
        pred (np.array): Predicted values

    Returns:
        float: RRMSE as a percentage
    """
    rmse = np.sqrt(np.square(gt - pred).mean())
    return 100 * (rmse / gt.mean())


def clean_column(df, column):
    """
    Cleans and interpolates missing values in a specified DataFrame column.
    In some cases the entries are just "++"

    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name to clean
    """
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].interpolate().fillna(method="bfill").fillna(method="ffill")


if __name__ == "__main__":
    # File paths
    # simulation_model_output_path = "/path/to/climateModel_hps_manuscriptParams.csv"
    simulation_model_output_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david/Simulation data/CSV output/climateModel_hps_manuscriptParams.csv"

    # gt_timeseries = "/path/to/greenlight_gt_timeseries.csv"
    gt_timeseries = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/greenlight_gt_timeseries.csv"

    # gt_hps_raw_path = "/path/to/HPS_raw.csv"
    gt_hps_raw_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david/Raw data/HPS raw.csv"

    # Load data
    gt_hps_raw = pd.read_csv(gt_hps_raw_path)
    clean_column(gt_hps_raw, "Indoor temperature (°C)")
    clean_column(gt_hps_raw, "Indoor relative humidity (%)")

    # Preparing ground truth vapor pressure
    gt_hps_raw_rh = np.array(gt_hps_raw["Indoor relative humidity (%)"]) / 100
    gt_hps_raw_temp = np.array(gt_hps_raw["Indoor temperature (°C)"])
    gt_hps_raw_vp_air = vapor_rh2pres(gt_hps_raw_temp, gt_hps_raw_rh)

    # Load simulation outputs
    david_simulation_model_output = pd.read_csv(simulation_model_output_path)
    gurkan_gt = pd.read_csv(gt_timeseries)

    # Calculate RRMSE for vapor pressure
    david_vp_air = np.array(david_simulation_model_output["Indoor vapor pressure (Pa)"])
    gurkan_gt_vp_air = np.array(gurkan_gt["vpAir"])
    rrmse_vp = calculate_rrmse(gurkan_gt_vp_air, david_vp_air)
    rrmse_vp_from_rh = calculate_rrmse(
        gt_hps_raw_vp_air[: len(david_vp_air)], david_vp_air
    )
    print("RRMSE Vapor Pressure:", rrmse_vp, rrmse_vp_from_rh)

    # Calculate RRMSE for temperature
    gurkan_gt_t_air = gurkan_gt["tAir"].to_numpy()
    david_t_air = david_simulation_model_output["Indoor temperature (°C)"].to_numpy()
    rrmse_t = calculate_rrmse(gurkan_gt_t_air, david_t_air)
    print("RRMSE Indoor Temperature:", rrmse_t)

    # Calculate RRMSE for relative humidity
    david_rh_air = np.array(
        david_simulation_model_output["Indoor relative humidity (%)"]
    )
    gurkan_gt_rh_air = 100 * gurkan_gt_vp_air / sat_vp(gurkan_gt_t_air)
    rrmse_rh = calculate_rrmse(gurkan_gt_rh_air, david_rh_air)
    print("RRMSE Relative Humidity:", rrmse_rh)

    # Calculate RRMSE for CO2 concentration
    gurkan_gt_co2_air = gurkan_gt["co2Air"].to_numpy()
    david_co2_air = david_simulation_model_output[
        "Indoor CO2 concentration (mg m^{-3})"
    ].to_numpy()
    rrmse_co2 = calculate_rrmse(gurkan_gt_co2_air, david_co2_air)
    print("RRMSE CO2 Concentration:", rrmse_co2)
