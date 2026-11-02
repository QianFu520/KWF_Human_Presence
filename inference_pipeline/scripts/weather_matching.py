import pandas as pd
import config


def add_weather_data(df_metadata, df_weather):
    df_metadata = df_metadata.copy()
    df_weather = df_weather.copy()

    # Round each clip's Timestamp (YYYYMMDD_HHMMSS, local time) to the nearest hour,
    # as in the training data (12:51 -> 13:00). Both sides are parsed datetimes, so the
    # merge does not depend on how the weather CSV formats its Datetime strings.
    df_metadata['Datetime'] = pd.to_datetime(
        df_metadata['Timestamp'].astype(str), format='%Y%m%d_%H%M%S'
    ).dt.round('h')
    df_weather['Datetime'] = pd.to_datetime(df_weather['Datetime'])

    # Merge metadata and weather so all weather is applied to specific times
    df_merged = df_metadata.merge(df_weather, on='Datetime', how='left')

    # Blank weather means no matching hour, or the API returned no value for that hour.
    # Stop here rather than pass blank features to the HP model.
    missing = df_merged[['Temperature', 'Windspeed', 'Humidity']].isna().any(axis=1)
    if missing.any():
        examples = df_merged.loc[missing, 'clip_name'].head(5).tolist()
        raise ValueError(
            f"{missing.sum()} of {len(df_merged)} clips have blank Temperature, Windspeed or "
            f"Humidity after matching weather, e.g. {examples}"
        )

    # Return just the clip names, and relevant weather data
    cols = df_weather.columns.tolist()
    cols.append('clip_name')

    return df_merged[cols]


def weather_match(weather_csv=config.WEATHER_DATA_PATH,
                  metadata_csv=config.METADATA_OUTPUT_CSV,
                  output_csv=config.WEATHER_OUTPUT_CSV):
    # Load weather data, metadata
    df_weather = pd.read_csv(weather_csv)
    df_metadata = pd.read_csv(metadata_csv)

    # Get matches, save
    df_matches = add_weather_data(df_metadata, df_weather)
    df_matches.to_csv(output_csv, index=False)
    return df_matches


if __name__ == "__main__":
    weather_match()
