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
    """Load cleaned operations and project metadata."""

    operations = pd.read_csv(
        PROCESSED_DATA_DIR
        / "daily_operations_clean.csv",
        parse_dates=["date"],
    )

    projects = pd.read_csv(
        RAW_DATA_DIR / "projects.csv",
        parse_dates=[
            "start_date",
            "planned_end_date",
        ],
    )

    return operations, projects


# =========================================================
# MERGE PROJECT METADATA
# =========================================================

def add_project_metadata(
    operations: pd.DataFrame,
    projects: pd.DataFrame,
) -> pd.DataFrame:
    """Attach project-level information to daily records."""

    project_columns = [
        "project_id",
        "project_name",
        "project_type",
        "location",
        "planned_budget_usd",
        "planned_duration_days",
        "base_workers",
    ]

    df = operations.merge(
        projects[project_columns],
        on="project_id",
        how="left",
        validate="many_to_one",
    )

    return df


# =========================================================
# PROGRESS FEATURES
# =========================================================

def add_progress_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create schedule and daily productivity features."""

    df = df.copy()

    df = df.sort_values(
        ["project_id", "date"]
    ).reset_index(drop=True)

    df["schedule_variance_pct"] = (
        df["actual_progress_pct"]
        - df["planned_progress_pct"]
    )

    df["daily_progress_gain"] = (
        df.groupby("project_id")[
            "actual_progress_pct"
        ]
        .diff()
    )

    first_project_row = (
        df.groupby("project_id").cumcount() == 0
    )

    df.loc[
        first_project_row,
        "daily_progress_gain",
    ] = df.loc[
        first_project_row,
        "actual_progress_pct",
    ]

    df["daily_progress_gain"] = (
        df["daily_progress_gain"]
        .clip(lower=0)
        .round(4)
    )

    return df


# =========================================================
# WORKFORCE FEATURES
# =========================================================

def add_workforce_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize workforce across projects."""

    df = df.copy()

    df["workforce_ratio"] = (
        df["workers"]
        / df["base_workers"]
    )

    df["workforce_variance_pct"] = (
        (
            df["workers"]
            - df["base_workers"]
        )
        / df["base_workers"]
        * 100
    )

    return df


# =========================================================
# FUEL FEATURES
# =========================================================

def add_fuel_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create fuel efficiency metrics."""

    df = df.copy()

    safe_equipment_hours = (
        df["equipment_hours"]
        .replace(0, np.nan)
    )

    df["fuel_per_equipment_hour"] = (
        df["fuel_consumption_l"]
        / safe_equipment_hours
    )

    return df


# =========================================================
# MATERIAL FEATURES
# =========================================================

def add_material_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create material delivery and efficiency metrics."""

    df = df.copy()

    safe_material_used = (
        df["material_used_t"]
        .replace(0, np.nan)
    )

    df["material_delivery_ratio"] = (
        df["material_delivered_t"]
        / safe_material_used
    )

    df["progress_per_material_t"] = (
        df["daily_progress_gain"]
        / safe_material_used
    )

    return df


# =========================================================
# COST FEATURES
# =========================================================

def add_cost_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create cost efficiency metrics."""

    df = df.copy()

    safe_progress = (
        df["daily_progress_gain"]
        .replace(0, np.nan)
    )

    df["cost_per_progress_point"] = (
        df["daily_cost_usd"]
        / safe_progress
    )

    return df


# =========================================================
# DELAY FEATURES
# =========================================================

def add_delay_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize daily delay severity."""

    df = df.copy()

    # Generator limits daily delay to 12 hours.
    df["delay_ratio"] = (
        df["delay_hours"] / 12
    )

    return df


# =========================================================
# ROLLING FEATURES
# =========================================================

def add_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create 7-day rolling operational indicators
    within each construction project.
    """

    df = df.copy()

    df["rolling_7d_fuel"] = (
        df.groupby("project_id")[
            "fuel_consumption_l"
        ]
        .transform(
            lambda series:
                series
                .rolling(
                    window=7,
                    min_periods=1,
                )
                .mean()
        )
    )

    df["rolling_7d_progress"] = (
        df.groupby("project_id")[
            "daily_progress_gain"
        ]
        .transform(
            lambda series:
                series
                .rolling(
                    window=7,
                    min_periods=1,
                )
                .mean()
        )
    )

    df["rolling_7d_delay"] = (
        df.groupby("project_id")[
            "delay_hours"
        ]
        .transform(
            lambda series:
                series
                .rolling(
                    window=7,
                    min_periods=1,
                )
                .mean()
        )
    )

    return df


# =========================================================
# SCHEDULE STATUS
# =========================================================

def add_schedule_status(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create descriptive schedule status categories.

    Thresholds are operational heuristics,
    not machine-learning predictions.
    """

    df = df.copy()

    conditions = [
        df["schedule_variance_pct"] < -2,
        df["schedule_variance_pct"] > 2,
    ]

    labels = [
        "Behind",
        "Ahead",
    ]

    df["schedule_status"] = np.select(
        conditions,
        labels,
        default="On Track",
    )

    df["schedule_risk_flag"] = (
        (
            df["schedule_variance_pct"] < -5
        )
        |
        (
            df["rolling_7d_delay"] >= 2.5
        )
    ).astype(int)

    return df


# =========================================================
# COMPLETE FEATURE PIPELINE
# =========================================================

def create_features(
    operations: pd.DataFrame,
    projects: pd.DataFrame,
) -> pd.DataFrame:
    """Run the complete SitePulse feature pipeline."""

    df = add_project_metadata(
        operations,
        projects,
    )

    df = add_progress_features(df)

    df = add_workforce_features(df)

    df = add_fuel_features(df)

    df = add_material_features(df)

    df = add_cost_features(df)

    df = add_delay_features(df)

    df = add_rolling_features(df)

    df = add_schedule_status(df)

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate_features(
    df: pd.DataFrame,
) -> None:
    """Validate engineered features."""

    required_features = [
        "schedule_variance_pct",
        "daily_progress_gain",
        "workforce_ratio",
        "fuel_per_equipment_hour",
        "material_delivery_ratio",
        "cost_per_progress_point",
        "delay_ratio",
        "rolling_7d_fuel",
        "rolling_7d_progress",
        "rolling_7d_delay",
        "schedule_status",
        "schedule_risk_flag",
    ]

    missing_features = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing engineered features: "
            f"{missing_features}"
        )

    if not df[
        "workforce_ratio"
    ].gt(0).all():
        raise ValueError(
            "Invalid workforce ratios detected."
        )

    if not df[
        "delay_ratio"
    ].between(0, 1).all():
        raise ValueError(
            "Delay ratio must be between 0 and 1."
        )

    valid_statuses = {
        "Behind",
        "On Track",
        "Ahead",
    }

    actual_statuses = set(
        df["schedule_status"].unique()
    )

    if not actual_statuses.issubset(
        valid_statuses
    ):
        raise ValueError(
            "Invalid schedule status detected."
        )


# =========================================================
# MAIN
# =========================================================

def main():
    """Execute SitePulse feature engineering."""

    operations, projects = load_data()

    features = create_features(
        operations,
        projects,
    )

    validate_features(features)

    output_path = (
        PROCESSED_DATA_DIR
        / "operations_features.csv"
    )

    features.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Rows: {len(features):,}"
    )

    print(
        f"Columns: {len(features.columns):,}"
    )

    print(
        "\nEngineered features:"
    )

    engineered_columns = [
        "schedule_variance_pct",
        "daily_progress_gain",
        "workforce_ratio",
        "workforce_variance_pct",
        "fuel_per_equipment_hour",
        "material_delivery_ratio",
        "progress_per_material_t",
        "cost_per_progress_point",
        "delay_ratio",
        "rolling_7d_fuel",
        "rolling_7d_progress",
        "rolling_7d_delay",
        "schedule_status",
        "schedule_risk_flag",
    ]

    for column in engineered_columns:
        print(f"- {column}")

    print(
        f"\nSaved feature dataset to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
