from typing import Tuple, Optional

import pandas as pd
import scipy.io
from datetime import datetime, timedelta


def matlab_datenum_to_datetime(matlab_datenum):
    # MATLAB datenum is the number of days since 1-Jan-0000
    python_datetime = (
        datetime.fromordinal(int(matlab_datenum))
        + timedelta(days=matlab_datenum % 1)
        - timedelta(days=366)
    )
    return python_datetime


def combine_climate_data(
    outdoor_df: pd.DataFrame, indoor_df: pd.DataFrame, controls_df: pd.DataFrame
) -> pd.DataFrame:
    combined_df = pd.merge(outdoor_df, indoor_df, on="time", how="outer")
    combined_df = pd.merge(combined_df, controls_df, on="time", how="outer")
    combined_df = combined_df.sort_values(by="time").reset_index(drop=True)
    return combined_df


def combine_gt_with_crop_and_aux_data(
    gt_df: pd.DataFrame, crop_df: Optional[pd.DataFrame], aux_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    # Ensure 'time' columns are not modified in gt_df, only merging by index
    gt_time = gt_df["time"]  # Store the original time column from gt_df

    if crop_df is not None:
        crop_df = crop_df.drop(
            columns=["time"], errors="ignore"
        )  # Drop the 'time' column from crop_df
        gt_df = pd.merge(gt_df, crop_df, left_index=True, right_index=True, how="outer")

    if aux_df is not None:
        aux_df = aux_df.drop(
            columns=["time"], errors="ignore"
        )  # Drop the 'time' column from aux_df
        gt_df = pd.merge(gt_df, aux_df, left_index=True, right_index=True, how="outer")

    # Restore the original time column from gt_df
    gt_df["time"] = gt_time

    # Sort by time if any merging took place and reset index
    if crop_df is not None or aux_df is not None:
        gt_df = gt_df.sort_values(by="time").reset_index(drop=True)

    return gt_df


def read_climate_model_gt_data(
    gt_data_mat_file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, datetime]:
    """
    - From GreenLight
    % Output:
    %   outdoor         A 6 column matrix with the following columns:
    %       outdoor(:,1)    timestamps of the input [s] in regular intervals of 300, starting with 0
    %       outdoor(:,2)    radiation   [W m^{-2}]              outdoor global irradiation
    %       outdoor(:,3)    temperature [C]                    outdoor air temperature
    %       outdoor(:,4)    humidity    [kg m^{-3}]             outdoor vapor concentration
    %       outdoor(:,5)    co2         [kg{CO2} m^{-3}{air}]   outdoor CO2 concentration
    %       outdoor(:,6)    wind        [m s^{-1}]              outdoor wind speed
    %   indoor          A 4 column matrix with:
    %       indoor(:,1)     timestamps of the input [s] in regular intervals of 300, starting with 0
    %       indoor(:,2)     temperature [C]                    indoor air temperature
    %       indoor(:,3)     humidity    [kg m^{-3}]             indoor vapor concentration
    %       indoor(:,4)     co2         [ppm]                   indoor co2 concentration
    %   controls
    %       controls(:,1)     timestamps of the input [s] in regular intervals of 300, starting with 0
    %       controls(:,2)     Energy screen closure 			0-1 (1 is fully closed)
    %       controls(:,3)     Black out screen closure			0-1 (1 is fully closed)
    %       controls(:,4)     Average ventilation aperture		0-1 (1 is fully open)
    %       controls(:,5)     Pipe rail temperature 			C
    %       controls(:,6)     Grow pipes temperature 			C
    %       controls(:,7)     Toplight on/off                   0/1 (1 is on)
    %       controls(:,8)     Interlights on/off                0/1 (1 is on)
    %       controls(:,9)     CO2 injection                     0/1 (1 is on)
    %       controls(:,10) 		Lee side ventilation aperture		0-1 (1 is fully open)
    %       controls(:,11) 		Wind side ventilation aperture 		0-1 (1 is fully open)

    %   startTime       date and time of starting point (datetime)

    - Formatted Output Naming
    #### 1. Outdoor Data
    The `outdoor` data, after manipulation, contains the following columns:

    | Column | GreenLight Parameter | Description                                   | Unit       |
    |--------|----------------------|-----------------------------------------------|------------|
    | 1      | `time`               | Timestamp (in seconds)                        | s          |
    | 2      | `iGlob`              | Outdoor global irradiation                    | W m⁻²      |
    | 3      | `tOut`               | Outdoor air temperature                       | °C         |
    | 4      | `vpOut`              | Outdoor vapor concentration (humidity)        | kg m⁻³     |
    | 5      | `co2Out`             | Outdoor CO₂ concentration                     | kg m⁻³     |
    | 6      | `wind`               | Outdoor wind speed                            | m s⁻¹      |
    | 7      | `tSky`               | Sky temperature (added by `skyTempRdam`)      | °C         |
    | 8      | `tSoOut`             | Soil temperature (added by `soilTempNl`)      | °C         |

    #### 2. Indoor Data
    The `indoor` data contains the following columns, after the conversion from vapor density to vapor pressure and CO₂ ppm to mg/m³:

    | Column | GreenLight Parameter | Description                                   | Unit       |
    |--------|----------------------|-----------------------------------------------|------------|
    | 1      | `time`               | Timestamp (in seconds)                        | s          |
    | 2      | `tAir`               | Indoor air temperature                        | °C         |
    | 3      | `vpAir`              | Indoor vapor pressure (converted)             | Pa         |
    | 4      | `co2Air`             | Indoor CO₂ concentration (converted)          | mg m⁻³     |

    #### 3. Controls Data
    The `controls` data contains control parameters like screen closures, pipe temperatures, and ventilation states:

    | Column | GreenLight Parameter | Description                                   | Unit       |
    |--------|----------------------|-----------------------------------------------|------------|
    | 1      | `time`               | Timestamp (in seconds)                        | s          |
    | 2      | `shScr`              | Energy screen closure (0-1)                   | -          |
    | 3      | `blScr`              | Blackout screen closure (0-1)                 | -          |
    | 4      | `roof`               | Average ventilation aperture (0-1)            | -          |
    | 5      | `tPipe`              | Pipe rail temperature                         | °C         |
    | 6      | `tGroPipe`           | Grow pipes temperature                        | °C         |
    | 7      | `lamp`               | Toplights on/off (0/1)                        | -          |
    | 8      | `intLamp`            | Interlights on/off (0/1)                      | -          |
    | 9      | `extCo2`             | CO₂ injection on/off (0/1)                    | -          |
    | 10     | `sideLee`            | Lee side ventilation aperture (0-1)           | -          |
    | 11     | `sideWind`           | Wind side ventilation aperture (0-1)          | -          |
    """
    data = scipy.io.loadmat(gt_data_mat_file_path)

    # Access the variables
    outdoor = data["outdoor"]
    indoor = data["indoor"]
    controls = data["controls"]
    # Access the serial number (datenum)
    start_time_serial = data["startTimeSerial"][0][0]

    # Convert start_time to Python datetime if necessary
    # start_time should be: 19-Oct-2009 15:15:00
    start_time = matlab_datenum_to_datetime(start_time_serial)
    print("Start Time:", start_time)

    # Create DataFrame for outdoor data
    outdoor_df = pd.DataFrame(
        outdoor,
        columns=["time", "iGlob", "tOut", "vpOut", "co2Out", "wind", "tSky", "tSoOut"],
    )

    # Create DataFrame for indoor data
    indoor_df = pd.DataFrame(indoor, columns=["time", "tAir", "vpAir", "co2Air"])

    # Create DataFrame for controls data
    controls_df = pd.DataFrame(
        controls,
        columns=[
            "time",
            "shScr",
            "blScr",
            "roof",
            "tPipe",
            "tGroPipe",
            "lamp",
            "intLamp",
            "extCo2",
            "sideLee",
            "sideWind",
        ],
    )

    return outdoor_df, indoor_df, controls_df, start_time


def read_climate_model_simulation_csv_data(
    file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads the climate model CSV data and formats it into outdoor, indoor, and controls DataFrames.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Outdoor, Indoor, and Controls DataFrames.
    """

    # Read CSV data
    data = pd.read_csv(file_path)

    # Create DataFrame for outdoor data
    # TODO: @gsoykan - since these are external variables we DO NOT need to REconsider them? (they are part of GT)
    outdoor_df = data[
        [
            "Time",  # Timestamp (in seconds) - TODO: @gsoykan - needs scaling
            "Global solar radiation (W m^{-2})",  # iGlob
            "Outdoor temperature (°C)",  # tOut
            "Outdoor vapor pressure (Pa)",  # vpOut - TODO: @gsoykan - needs scaling
            "Outdoor CO2 concentration (mg m^{-3})",  # co2Out - TODO: @gsoykan - needs scaling
            "Outdoor wind speed (m s^{-1})",  # wind
            "Apparent sky temperature (°C)",  # tSky
            "Temperature of external soil layer (°C)",  # tSoOut
        ]
    ].copy()
    outdoor_df.columns = [
        "time",
        "iGlob",
        "tOut",
        "vpOut",
        "co2Out",
        "wind",
        "tSky",
        "tSoOut",
    ]

    # Create DataFrame for indoor data
    # these have the same scale...
    indoor_df = data[
        [
            "Time",  # Timestamp (in seconds)
            "Indoor temperature (°C)",  # tAir
            "Indoor vapor pressure (Pa)",  # vpAir
            "Indoor CO2 concentration (mg m^{-3})",  # co2Air
        ]
    ].copy()
    indoor_df.columns = ["time", "tAir", "vpAir", "co2Air"]

    # Create DataFrame for controls data
    controls_df = data[
        [
            "Time",  # Timestamp (in seconds)
            "Shading screen position (0-1)",  # shScr
            "Blackout screen position (0-1)",  # blScr
            "Roof ventilation position (0-1)",  # roof
            "Pipe rail temperature (°C)",  # tPipe
            "Grow pipe temperature (°C)",  # tGroPipe
            "Lamp status (0-1)",  # lamp
            "Interlamp status (0-1)",  # intLamp
            "CO2 injection valve position (0-1)",  # extCo2
            # "Lee side ventilation aperture (0-1)",  # sideLee - TODO: @gsoykan - does not exist
            # "Wind side ventilation aperture (0-1)",  # sideWind - TODO: @gsoykan - does not exist
        ]
    ].copy()
    controls_df.columns = [
        "time",
        "shScr",
        "blScr",
        "roof",
        "tPipe",
        "tGroPipe",
        "lamp",
        "intLamp",
        "extCo2",
        # "sideLee",
        # "sideWind",
    ]

    # Create DataFrame for crop model data
    crop_df = data[
        [
            "Time",  # Timestamp (in seconds)
            "Crop development stage (°C day)",  # tCanSum - might needs scaling
            "Carbohydrates in buffer (mg{CH2O} m^{-2})",  # cBuf - needs scaling
            "Carbohydrates in leaves (mg{CH2O} m^{-2})",  # cLeaf - needs scaling
            "Carbohydrates in stems (mg{CH2O} m^{-2})",  # cStem - needs scaling
            "Carbohydrates in fruits (mg{CH2O} m^{-2})",  # cFruit - needs scaling
            "Leaf area index (m^2 {leaf} m^{-2} {floor})",  # lai
        ]
    ].copy()
    crop_df.columns = [
        "time",
        "tCanSum",
        "cBuf",
        "cLeaf",
        "cStem",
        "cFruit",
        "lai",
    ]

    # Create DataFrame for auxiliary states data
    # simulation csv headers:
    # Time	Outdoor temperature (°C)	Outdoor vapor pressure (Pa)	Outdoor CO1 concentration (mg m^{-3})	Outdoor wind speed (m s^{-1})	Apparent sky temperature (°C)	Temperature of external soil layer (°C)	Global solar radiation (W m^{-2})	Indoor temperature (°C)	Indoor vapor pressure (Pa)	Indoor relative humidity (%)	Indoor CO2 concentration (mg m^{-3})	Indoor CO2 concentration (ppm)	Global radiation above the canopy (W m^{-2})	Temperature above the screen (°C)	Vapor pressure above the screen (Pa)	CO2 concentration above the screen (mg m^{-3})	External cover temperature (°C)	Internal cover temperature (°C)	Thermal screen temperature (°C)	Blackout screen temperature (°C)	Lamp temperature (°C)	Interlamp temperature (°C)	Canopy temperature (°C)	Average 24 hour canopy temperature (°C)	Pipe rail temperature (°C)	Grow pipe temperature (°C)	Floor temperature (°C)	Soil layer 1 temperature (°C)	Soil layer 2 temperature (°C)	Soil layer 3 temperature (°C)	Soil layer 4 temperature (°C)	Soil layer 5 temperature (°C)	Heating setpoint (°C)	Cooling setpoint (°C)	CO2 setpoint (ppm)	Relative humidity setpoint (%)	Pipe rail boiler valve position (0-1)	Grow pipe boiler valve position (0-1)	Shading screen position (0-1)	Thermal screen position (0-1)	Blackout screen position (0-1)	Roof ventilation position (0-1)	Lamp status (0-1)	Interlamp status (0-1)	CO2 injection valve position (0-1)	Energy supply from the pipe rails (W m^{-2})	Energy supply from the grow pipes (W m^{-2})	Energy supply from the lamps (W m^{-2})	Energy supply from the interlamps (W m^{-2})	Lamp cooling (W m^{-2})	Ventilation rate (m^{3} m^{-2} s^{-1}	CO2 injection rate (mg m^{-2} s^{-1})	Crop development stage (°C day)	Carbohydrates in buffer (mg{CH2O} m^{-2})	Carbohydrates in leaves (mg{CH2O} m^{-2})	Carbohydrates in stems (mg{CH2O} m^{-2})	Carbohydrates in fruits (mg{CH2O} m^{-2})	Leaf area index (m^2 {leaf} m^{-2} {floor})	Net photosynthesis (mg{CH2O} m^{-2} s^{-1})	Carboyhdrate flow from buffer to leaves (mg{CH2O} m^{-2})	Carboyhdrate flow from buffer to frtuis (mg{CH2O} m^{-2})	Carboyhdrate flow from buffer to stems (mg{CH2O} m^{-2})	Growth respiration (mg{CH2O} m^{-2})	Leaf maintenance respiration (mg{CH2O} m^{-2})	Fruit maintenance respiration (mg{CH2O} m^{-2})	Stem maintenance respiration (mg{CH2O} m^{-2})	Leaf pruning (mg{CH2O} m^{-2})	Fruit harvest (mg{CH2O} m^{-2})	Net crop assimilation (mg{CH2O} m^{-2})	Canopy transpiration (kg m^{-2} s^{-1})
    aux_states_df = data[
        [
            "Time",  # Timestamp (in seconds)
            "Energy supply from the pipe rails (W m^{-2})",  # qRail
            "Energy supply from the grow pipes (W m^{-2})",  # qGro
            "Energy supply from the lamps (W m^{-2})",  # lampIn
            "Energy supply from the interlamps (W m^{-2})",  # intLampIn
            "Lamp cooling (W m^{-2})",  # lampCool
            "Ventilation rate (m^{3} m^{-2} s^{-1}",  # vent
            "CO2 injection rate (mg m^{-2} s^{-1})",  # co2Inj
            # above climate control aux. states
            "Net photosynthesis (mg{CH2O} m^{-2} s^{-1})",  # mcAirBuf
            "Carboyhdrate flow from buffer to leaves (mg{CH2O} m^{-2})",  # mcBufLeaf
            "Carboyhdrate flow from buffer to frtuis (mg{CH2O} m^{-2})",  # mcBufFruit
            "Carboyhdrate flow from buffer to stems (mg{CH2O} m^{-2})",  # mcBufStem
            "Growth respiration (mg{CH2O} m^{-2})",  # mcBufAir
            "Leaf maintenance respiration (mg{CH2O} m^{-2})",  # mcLeafAir
            "Fruit maintenance respiration (mg{CH2O} m^{-2})",  # mcFruitAir
            "Stem maintenance respiration (mg{CH2O} m^{-2})",  # mcStemAir
            "Leaf pruning (mg{CH2O} m^{-2})",  # mcLeafHar
            "Fruit harvest (mg{CH2O} m^{-2})",  # mcFruitHar
            "Net crop assimilation (mg{CH2O} m^{-2})",  # mcAirCan
            "Canopy transpiration (kg m^{-2} s^{-1})",  # mvCanAir
            # above crop aux. states
        ]
    ].copy()

    aux_states_df.columns = [
        "time",
        "qRail",
        "qGro",
        "lampIn",
        "intLampIn",
        "lampCool",
        "vent",
        "co2Inj",  # climate control aux. states
        "mcAirBuf",
        "mcBufLeaf",
        "mcBufFruit",
        "mcBufStem",
        "mcBufAir",
        "mcLeafAir",
        "mcFruitAir",
        "mcStemAir",
        "mcLeafHar",
        "mcFruitHar",
        "mcAirCan",
        "mvCanAir",  # crop aux. states
    ]

    return outdoor_df, indoor_df, controls_df, crop_df, aux_states_df


def create_i_transformers_dataset(
    gt_data_mat_file_path: str = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/climate_model_hps_gt_data.mat",
    output_csv_path: str = "greenlight_gt_itransformers.csv",
):
    outdoor_df, indoor_df, controls_df, start_time = read_climate_model_gt_data(
        gt_data_mat_file_path
    )
    combined_df = combine_climate_data(outdoor_df, indoor_df, controls_df)

    # 19-Oct-2009 15:15:00
    # Round to the nearest minute
    rounded_time = (start_time + timedelta(seconds=30)).replace(second=0, microsecond=0)

    # Update 'time' column to represent the actual timestamps
    combined_df["time"] = combined_df["time"].apply(
        lambda x: rounded_time + timedelta(seconds=x)
    )
    # 2020-01-01 00:10:00
    # Convert to desired string format
    combined_df["time"] = combined_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    combined_df.rename(columns={"time": "date"}, inplace=True)
    combined_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    # simulation_data_csv_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david/Simulation data/CSV output/climateModel_hps_manuscriptParams.csv"
    # outdoor_sim_df, indoor_sim_df, controls_sim_df, crop_df, aux_states_df = (
    #     read_climate_model_simulation_csv_data(simulation_data_csv_path)
    # )

    create_i_transformers_dataset(gt_data_mat_file_path="/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/gt/led/climate_model_led_gt_data.mat",
                                  output_csv_path="/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/gt/led/gt_led_timeseries.csv")

    gt_data_mat_file_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/climate_model_hps_gt_data.mat"
    outdoor_df, indoor_df, controls_df, start_time = read_climate_model_gt_data(
        gt_data_mat_file_path
    )
    print(outdoor_df, indoor_df, controls_df)
