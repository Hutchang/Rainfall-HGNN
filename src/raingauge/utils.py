import pandas as pd
from pathlib import Path


def load_raingauge_dataset(
    start:int,
    end: int,
    uptime_threshold: float = 1.0,
    folder_path: str = 'database/sydney_gauge_legacy',
) -> tuple:
    """Load the transitional Sydney hourly gauge export without rescaling it.

    This loader exists only for the inherited gauge-only smoke runner. Components 01
    and 02 will provide the production loaders and timezone-aware UTC data.
    """

    folder = Path(folder_path)

    #Process raingauge df
    print(f"Loading raingauge data from {start} to {end}")
    complete_raingauge_df = []
    for year in range(start, end + 1):
        filepath = folder / str(year) / f"weather_station_data_{year}.csv"
        print(f"Loading raingauge_dataset from {filepath}")
        raingauge_df = pd.read_csv(filepath)
        complete_raingauge_df.append(raingauge_df)

    gauge_df = pd.concat(complete_raingauge_df)


    gauge_df["timestamp"] = pd.to_datetime(
        gauge_df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    formatted_gauge_df = gauge_df.pivot(
        index="timestamp", columns="stationId", values="value"
    )
    print("Loading complete")
    print(f"Dataframe shape: {formatted_gauge_df.shape}")

    #Filter for threshold
    filtered_stations_index = formatted_gauge_df.columns
    if uptime_threshold is not None:
        print(f"Filtering raingauge_uptime. Threshold = {uptime_threshold}")
        filtered_stations_index = filter_uptime(formatted_gauge_df, uptime_threshold=uptime_threshold).index
        formatted_gauge_df = formatted_gauge_df[filtered_stations_index]

    #Get raingauge coordinate mappings
    print("Getting raingauge coordinate mappings")
    raingauge_mappings_df = get_station_mapping_df(
        start=start, end=end, folder_path=folder
    )
    raingauge_mappings_df = raingauge_mappings_df[raingauge_mappings_df['id'].isin(filtered_stations_index)]
    print(f"Mapping df shape: {raingauge_mappings_df.shape}")
    raingauge_mappings_df['order'] = [i for i in range(raingauge_mappings_df.shape[0])]
    raingauge_mappings_df.reset_index(inplace=True)

    return formatted_gauge_df, raingauge_mappings_df



def filter_uptime(raingauge_df: pd.DataFrame, uptime_threshold = 0.9) -> pd.DataFrame:
    '''
    Filters dataframe for threshold where we keep only stations with >threshold uptime

    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    raingauge_uptime = raingauge_df.notna().sum() / len(raingauge_df)
    filtered_stations_df = raingauge_uptime[raingauge_uptime >= uptime_threshold]
    return filtered_stations_df



def get_station_coordinate_mappings(
    start: int,
    end: int,
    folder_path: str | Path = "database/sydney_gauge_legacy",
) -> dict:
    """
    Returns dictionary containing the mappings of station names to coordinates for raingauge

    dict: [key, (lat,lon)]
    ------
    """

    station_df = pd.DataFrame()

    folder = Path(folder_path)
    for year in range(start, end + 1):
        df = pd.read_csv(folder / str(year) / f"weather_stations_{year}.csv")
        station_df = pd.concat([station_df, df]).drop_duplicates(['id', 'latitude', 'longitude']).reset_index(drop=True)
    station_dict = dict(zip(station_df['id'], zip(station_df['latitude'], station_df['longitude'])))
    return station_dict



def get_station_mapping_df(
    start: int,
    end: int,
    folder_path: str | Path = "database/sydney_gauge_legacy",
) -> pd.DataFrame:
    station_df = pd.DataFrame()

    folder = Path(folder_path)
    for year in range(start, end + 1):
        df = pd.read_csv(folder / str(year) / f"weather_stations_{year}.csv")
        station_df = pd.concat([station_df, df]).drop_duplicates(['id', 'latitude', 'longitude']).reset_index(drop=True)

    station_df.drop(columns='deviceId', inplace=True)
    return station_df
