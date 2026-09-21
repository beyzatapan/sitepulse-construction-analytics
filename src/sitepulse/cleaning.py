from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    """Load raw SitePulse datasets."""

    projects = pd.read_csv(
        RAW_DATA_DIR / "projects.csv"
    )

    equipment = pd.read_csv(
        RAW_DATA_DIR / "equipment.csv"
    )

    operations = pd.read_csv(
        RAW_DATA_DIR / "daily_operations.csv"
    )

    return projects, equipment, operations


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove exact duplicate operational records."""

    return (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )


# =========================================================
# DATE CONVERSION
# =========================================================

def convert_dates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert date column to pandas datetime."""

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    return df


# =========================================================
# WORKFORCE IMPUTATION
# =========================================================

def impute_workers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing workforce values using
    the median workforce of the same project.
    """

    df = df.copy()

    project_median = (
        df.groupby("project_id")["workers"]
        .transform("median")
    )

    global_median = df["workers"].median()

    df["workers"] = (
        df["workers"]
        .fillna(project_median)
        .fillna(global_median)
        .round()
        .astype("Int64")
    )

    return df


# =========================================================
# FUEL IMPUTATION
# =========================================================

def impute_fuel_consumption(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate missing fuel consumption from
    equipment hours and normal project-level
    fuel consumption rates.
    """

    df = df.copy()

    valid_rate_mask = (
        df["fuel_consumption_l"].notna()
        & df["equipment_hours"].gt(0)
        & df["fuel_anomaly_flag"].eq(0)
    )

    fuel_rate = (
        df["fuel_consumption_l"]
        / df["equipment_hours"].replace(0, np.nan)
    )

    normal_fuel_rate = fuel_rate.where(
        valid_rate_mask
    )

    project_rate = (
        normal_fuel_rate
        .groupby(df["project_id"])
        .transform("median")
    )

    global_rate = normal_fuel_rate.median()

    missing_mask = (
        df["fuel_consumption_l"].isna()
    )

    estimated_fuel = (
        df["equipment_hours"]
        * project_rate.fillna(global_rate)
    )

    df.loc[
        missing_mask,
        "fuel_consumption_l",
    ] = estimated_fuel[missing_mask]

    return df


# =========================================================
# MATERIAL DELIVERY IMPUTATION
# =========================================================

def impute_material_delivery(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate missing material deliveries using
    project and shortage-specific delivery ratios.
    """

    df = df.copy()

    material_used = (
        df["material_used_t"]
        .replace(0, np.nan)
    )

    delivery_ratio = (
        df["material_delivered_t"]
        / material_used
    )

    group_ratio = (
        delivery_ratio
        .groupby(
            [
                df["project_id"],
                df["material_shortage_flag"],
            ]
        )
        .transform("median")
    )

    global_ratio = delivery_ratio.median()

    missing_mask = (
        df["material_delivered_t"].isna()
    )

    estimated_delivery = (
        df["material_used_t"]
        * group_ratio.fillna(global_ratio)
    )

    df.loc[
        missing_mask,
        "material_delivered_t",
    ] = estimated_delivery[missing_mask]

    return df


# =========================================================
# WEATHER IMPUTATION
# =========================================================

def impute_weather(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing weather using the most common
    weather condition for the same project/month.
    """

    df = df.copy()

    df["_month"] = df["date"].dt.month

    def mode_or_missing(series):
        mode = series.mode()

        if mode.empty:
            return pd.NA

        return mode.iloc[0]

    project_month_mode = (
        df.groupby(
            ["project_id", "_month"]
        )["weather"]
        .transform(mode_or_missing)
    )

    global_mode = (
        df["weather"]
        .mode()
        .iloc[0]
    )

    df["weather"] = (
        df["weather"]
        .fillna(project_month_mode)
        .fillna(global_mode)
    )

    df = df.drop(
        columns="_month"
    )

    return df


# =========================================================
# ROUND NUMERIC VALUES
# =========================================================

def round_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep operational measurements consistent."""

    df = df.copy()

    columns_to_round = [
        "equipment_hours",
        "fuel_consumption_l",
        "material_delivered_t",
        "material_used_t",
        "planned_progress_pct",
        "actual_progress_pct",
        "daily_cost_usd",
        "delay_hours",
    ]

    df[columns_to_round] = (
        df[columns_to_round]
        .round(2)
    )

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate_operations(
    df: pd.DataFrame,
    projects: pd.DataFrame,
) -> None:
    """
    Validate core business and data-quality rules.

    Raises ValueError if a rule is violated.
    """

    if df.duplicated().any():
        raise ValueError(
            "Duplicate rows remain in cleaned data."
        )

    if df.isna().any().any():
        raise ValueError(
            "Missing values remain in cleaned data."
        )

    if not df["workers"].gt(0).all():
        raise ValueError(
            "Workers must be greater than zero."
        )

    if not df["equipment_hours"].ge(0).all():
        raise ValueError(
            "Equipment hours cannot be negative."
        )

    if not df["fuel_consumption_l"].ge(0).all():
        raise ValueError(
            "Fuel consumption cannot be negative."
        )

    if not df["material_used_t"].ge(0).all():
        raise ValueError(
            "Material usage cannot be negative."
        )

    if not df[
        "material_delivered_t"
    ].ge(0).all():
        raise ValueError(
            "Material delivery cannot be negative."
        )

    if not df[
        "planned_progress_pct"
    ].between(0, 100).all():
        raise ValueError(
            "Planned progress must be between 0 and 100."
        )

    if not df[
        "actual_progress_pct"
    ].between(0, 100).all():
        raise ValueError(
            "Actual progress must be between 0 and 100."
        )

    if not df["delay_hours"].ge(0).all():
        raise ValueError(
            "Delay hours cannot be negative."
        )

    valid_flags = {0, 1}

    flag_columns = [
        "material_shortage_flag",
        "equipment_breakdown_flag",
        "fuel_anomaly_flag",
    ]

    for column in flag_columns:

        values = set(
            df[column].dropna().unique()
        )

        if not values.issubset(valid_flags):
            raise ValueError(
                f"Invalid values detected in {column}."
            )

    valid_projects = set(
        projects["project_id"]
    )

    operation_projects = set(
        df["project_id"]
    )

    invalid_projects = (
        operation_projects
        - valid_projects
    )

    if invalid_projects:
        raise ValueError(
            "Unknown project IDs detected: "
            f"{invalid_projects}"
        )


# =========================================================
# CLEANING PIPELINE
# =========================================================

def clean_operations(
    operations: pd.DataFrame,
) -> pd.DataFrame:
    """Run the complete operations cleaning pipeline."""

    df = operations.copy()

    df = remove_duplicates(df)

    df = convert_dates(df)

    df = impute_workers(df)

    df = impute_fuel_consumption(df)

    df = impute_material_delivery(df)

    df = impute_weather(df)

    df = round_numeric_columns(df)

    df = df.sort_values(
        ["project_id", "date"]
    ).reset_index(drop=True)

    return df


# =========================================================
# MAIN
# =========================================================

def main():
    """Execute SitePulse cleaning pipeline."""

    projects, equipment, operations = (
        load_data()
    )

    print(
        f"Raw operations rows: "
        f"{len(operations):,}"
    )

    print(
        f"Raw duplicate rows: "
        f"{operations.duplicated().sum():,}"
    )

    print(
        f"Raw missing values: "
        f"{operations.isna().sum().sum():,}"
    )

    cleaned_operations = clean_operations(
        operations
    )

    validate_operations(
        cleaned_operations,
        projects,
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "daily_operations_clean.csv"
    )

    cleaned_operations.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nCleaned operations rows: "
        f"{len(cleaned_operations):,}"
    )

    print(
        f"Cleaned duplicate rows: "
        f"{cleaned_operations.duplicated().sum():,}"
    )

    print(
        f"Cleaned missing values: "
        f"{cleaned_operations.isna().sum().sum():,}"
    )

    print(
        f"\nSaved cleaned dataset to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()