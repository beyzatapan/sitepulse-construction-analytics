from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# PROJECT DEFINITIONS
# =========================================================

PROJECTS = [
    {
        "project_id": "PRJ001",
        "project_name": "Northgate Industrial Park",
        "project_type": "Industrial",
        "location": "Abuja",
        "planned_budget_usd": 6_500_000,
        "planned_duration_days": 240,
        "start_date": "2026-01-05",
        "base_workers": 85,
        "productivity_multiplier": 0.96,

    },
    {
        "project_id": "PRJ002",
        "project_name": "Riverside Secondary School",
        "project_type": "Education",
        "location": "Abuja",
        "planned_budget_usd": 3_200_000,
        "planned_duration_days": 210,
        "start_date": "2026-01-19",
        "base_workers": 55,
        "productivity_multiplier": 1.08,
    },
    {
        "project_id": "PRJ003",
        "project_name": "Central Logistics Warehouse",
        "project_type": "Warehouse",
        "location": "Kaduna",
        "planned_budget_usd": 4_100_000,
        "planned_duration_days": 190,
        "start_date": "2026-02-02",
        "base_workers": 65,
        "productivity_multiplier": 1.14,
    },
    {
        "project_id": "PRJ004",
        "project_name": "Westlink Road Expansion",
        "project_type": "Infrastructure",
        "location": "Abuja",
        "planned_budget_usd": 8_800_000,
        "planned_duration_days": 270,
        "start_date": "2026-01-12",
        "base_workers": 65,
        "productivity_multiplier": 1.14,
    },
    {
        "project_id": "PRJ005",
        "project_name": "Greenfield Residential Estate",
        "project_type": "Residential",
        "location": "Lagos",
        "planned_budget_usd": 5_750_000,
        "planned_duration_days": 250,
        "start_date": "2026-02-09",
        "base_workers": 95,
        "productivity_multiplier": 1.18,
    },
    {
        "project_id": "PRJ006",
        "project_name": "Metro Commercial Complex",
        "project_type": "Commercial",
        "location": "Abuja",
        "planned_budget_usd": 7_300_000,
        "planned_duration_days": 230,
        "start_date": "2026-01-26",
        "base_workers": 90,
        "productivity_multiplier": 1.12,
    },
]


# =========================================================
# EQUIPMENT DEFINITIONS
# =========================================================

EQUIPMENT_PROFILES = {
    "Excavator": {
        "fuel_type": "Diesel",
        "expected_fuel_per_hour": 18.0,
    },
    "Wheel Loader": {
        "fuel_type": "Diesel",
        "expected_fuel_per_hour": 15.0,
    },
    "Bulldozer": {
        "fuel_type": "Diesel",
        "expected_fuel_per_hour": 22.0,
    },
    "Motor Grader": {
        "fuel_type": "Diesel",
        "expected_fuel_per_hour": 16.0,
    },
    "Mobile Crane": {
        "fuel_type": "Diesel",
        "expected_fuel_per_hour": 13.0,
    },
    "Concrete Mixer": {
        "fuel_type": "Diesel",
        "expected_fuel_per_hour": 9.0,
    },
    "Generator": {
        "fuel_type": "Diesel",
        "expected_fuel_per_hour": 11.0,
    },
}


# =========================================================
# WEATHER EFFECTS
# =========================================================

WEATHER_PROBABILITIES = {
    "Clear": 0.54,
    "Cloudy": 0.23,
    "Light Rain": 0.15,
    "Heavy Rain": 0.08,
}

WEATHER_PRODUCTIVITY = {
    "Clear": 1.00,
    "Cloudy": 0.97,
    "Light Rain": 0.82,
    "Heavy Rain": 0.55,
}

WEATHER_DELAY_HOURS = {
    "Clear": 0.0,
    "Cloudy": 0.0,
    "Light Rain": 1.5,
    "Heavy Rain": 4.0,
}


# =========================================================
# CREATE PROJECT DATASET
# =========================================================

def create_projects_dataframe() -> pd.DataFrame:
    """Create the master project table."""

    df = pd.DataFrame(PROJECTS)

    df["start_date"] = pd.to_datetime(df["start_date"])

    df["planned_end_date"] = (
        df["start_date"]
        + pd.to_timedelta(df["planned_duration_days"], unit="D")
    )

    return df


# =========================================================
# CREATE EQUIPMENT DATASET
# =========================================================

def create_equipment_dataframe(
    projects_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create equipment assigned to each project."""

    equipment_rows = []

    equipment_counter = 1

    for _, project in projects_df.iterrows():

        number_of_equipment = int(
            rng.integers(4, 8)
        )

        equipment_types = rng.choice(
            list(EQUIPMENT_PROFILES.keys()),
            size=number_of_equipment,
            replace=False,
        )

        for equipment_type in equipment_types:

            profile = EQUIPMENT_PROFILES[equipment_type]

            equipment_rows.append(
                {
                    "equipment_id": f"EQ{equipment_counter:03d}",
                    "project_id": project["project_id"],
                    "equipment_type": equipment_type,
                    "fuel_type": profile["fuel_type"],
                    "expected_fuel_per_hour": (
                        profile["expected_fuel_per_hour"]
                    ),
                }
            )

            equipment_counter += 1

    return pd.DataFrame(equipment_rows)


# =========================================================
# CREATE DAILY OPERATIONS DATASET
# =========================================================

def create_daily_operations_dataframe(
    projects_df: pd.DataFrame,
    equipment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate realistic daily construction operations."""

    rows = []

    weather_names = list(
        WEATHER_PROBABILITIES.keys()
    )

    weather_probabilities = list(
        WEATHER_PROBABILITIES.values()
    )

    for _, project in projects_df.iterrows():

        project_id = project["project_id"]
        duration = int(
            project["planned_duration_days"]
        )

        project_equipment = equipment_df[
            equipment_df["project_id"] == project_id
        ]

        average_fuel_rate = (
            project_equipment[
                "expected_fuel_per_hour"
            ].mean()
        )

        dates = pd.date_range(
            start=project["start_date"],
            periods=duration,
            freq="D",
        )

        actual_progress = 0.0

        planned_daily_increment = (
            100 / duration
        )

        for day_number, current_date in enumerate(
            dates,
            start=1,
        ):

            # ---------------------------------------------
            # WEATHER
            # ---------------------------------------------

            weather = rng.choice(
                weather_names,
                p=weather_probabilities,
            )

            weather_factor = (
                WEATHER_PRODUCTIVITY[weather]
            )

            # ---------------------------------------------
            # WORKFORCE
            # ---------------------------------------------

            workers = int(
                max(
                    10,
                    rng.normal(
                        project["base_workers"],
                        project["base_workers"] * 0.10,
                    ),
                )
            )

            workforce_factor = (
                workers / project["base_workers"]
            )

            # ---------------------------------------------
            # MATERIAL SHORTAGE
            # ---------------------------------------------

            material_shortage = (
                rng.random() < 0.055
            )

            material_factor = (
                0.60
                if material_shortage
                else 1.00
            )

            # ---------------------------------------------
            # EQUIPMENT BREAKDOWN
            # ---------------------------------------------

            equipment_breakdown = (
                rng.random() < 0.035
            )

            equipment_factor = (
                0.70
                if equipment_breakdown
                else 1.00
            )

            # ---------------------------------------------
            # PRODUCTIVITY
            # ---------------------------------------------

            productivity_noise = rng.normal(
                1.0,
                0.08,
            )

            productivity_factor = (
                project["productivity_multiplier"]
                * weather_factor
                * workforce_factor
                * material_factor
                * equipment_factor
                * productivity_noise
            )

            actual_daily_increment = (
                planned_daily_increment
                * productivity_factor
            )

            actual_daily_increment = max(
                0,
                actual_daily_increment,
            )

            actual_progress = min(
                100,
                actual_progress
                + actual_daily_increment,
            )

            planned_progress = min(
                100,
                (
                    day_number
                    / duration
                )
                * 100,
            )

            # ---------------------------------------------
            # EQUIPMENT HOURS
            # ---------------------------------------------

            base_equipment_hours = (
                len(project_equipment) * 7.5
            )

            equipment_hours = (
                base_equipment_hours
                * weather_factor
                * equipment_factor
                * rng.normal(1.0, 0.08)
            )

            equipment_hours = max(
                0,
                equipment_hours,
            )

            # ---------------------------------------------
            # FUEL CONSUMPTION
            # ---------------------------------------------

            expected_fuel = (
                equipment_hours
                * average_fuel_rate
            )

            fuel_consumption = (
                expected_fuel
                * rng.normal(1.0, 0.07)
            )

            fuel_anomaly = (
                rng.random() < 0.025
            )

            if fuel_anomaly:
                fuel_consumption *= rng.uniform(
                    1.40,
                    1.90,
                )

            fuel_consumption = max(
                0,
                fuel_consumption,
            )

            # ---------------------------------------------
            # MATERIALS
            # ---------------------------------------------

            base_material_need = (
                actual_daily_increment
                * rng.uniform(18, 30)
            )

            material_used = max(
                0,
                base_material_need
                * rng.normal(1.0, 0.08),
            )

            if material_shortage:

                material_delivered = (
                    material_used
                    * rng.uniform(
                        0.35,
                        0.75,
                    )
                )

            else:

                material_delivered = (
                    material_used
                    * rng.uniform(
                        0.95,
                        1.35,
                    )
                )

            # ---------------------------------------------
            # DELAY HOURS
            # ---------------------------------------------

            delay_hours = (
                WEATHER_DELAY_HOURS[weather]
            )

            if material_shortage:
                delay_hours += rng.uniform(
                    2,
                    5,
                )

            if equipment_breakdown:
                delay_hours += rng.uniform(
                    2,
                    6,
                )

            delay_hours += max(
                0,
                rng.normal(0, 0.5),
            )

            delay_hours = min(
                delay_hours,
                12,
            )

            # ---------------------------------------------
            # DAILY COST
            # ---------------------------------------------

            labour_cost = (
                workers
                * rng.uniform(35, 55)
            )

            equipment_cost = (
                equipment_hours
                * rng.uniform(45, 75)
            )

            material_cost = (
                material_used
                * rng.uniform(70, 130)
            )

            delay_cost = (
                delay_hours
                * rng.uniform(250, 600)
            )

            daily_cost = (
                labour_cost
                + equipment_cost
                + material_cost
                + delay_cost
            )

            # ---------------------------------------------
            # STORE RECORD
            # ---------------------------------------------

            rows.append(
                {
                    "date": current_date,
                    "project_id": project_id,
                    "workers": workers,
                    "equipment_hours": round(
                        equipment_hours,
                        2,
                    ),
                    "fuel_consumption_l": round(
                        fuel_consumption,
                        2,
                    ),
                    "material_delivered_t": round(
                        material_delivered,
                        2,
                    ),
                    "material_used_t": round(
                        material_used,
                        2,
                    ),
                    "planned_progress_pct": round(
                        planned_progress,
                        2,
                    ),
                    "actual_progress_pct": round(
                        actual_progress,
                        2,
                    ),
                    "daily_cost_usd": round(
                        daily_cost,
                        2,
                    ),
                    "weather": weather,
                    "delay_hours": round(
                        delay_hours,
                        2,
                    ),
                    "material_shortage_flag": int(
                        material_shortage
                    ),
                    "equipment_breakdown_flag": int(
                        equipment_breakdown
                    ),
                    "fuel_anomaly_flag": int(
                        fuel_anomaly
                    ),
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# ADD REALISTIC DATA QUALITY ISSUES
# =========================================================

def inject_data_quality_issues(
    operations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add controlled missing values and duplicate rows.

    These issues are intentionally added so that later
    stages of the project can demonstrate data cleaning.
    """

    df = operations_df.copy()

    missing_columns = [
        "workers",
        "fuel_consumption_l",
        "material_delivered_t",
        "weather",
    ]

    for column in missing_columns:

        missing_count = max(
            1,
            int(len(df) * 0.01),
        )

        missing_indices = rng.choice(
            df.index,
            size=missing_count,
            replace=False,
        )

        df.loc[
            missing_indices,
            column,
        ] = np.nan

    duplicate_count = max(
        1,
        int(len(df) * 0.005),
    )

    duplicate_indices = rng.choice(
        df.index,
        size=duplicate_count,
        replace=False,
    )

    duplicated_rows = df.loc[
        duplicate_indices
    ].copy()

    df = pd.concat(
        [
            df,
            duplicated_rows,
        ],
        ignore_index=True,
    )

    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return df


# =========================================================
# VALIDATION SUMMARY
# =========================================================

def print_validation_summary(
    projects_df: pd.DataFrame,
    equipment_df: pd.DataFrame,
    operations_df: pd.DataFrame,
) -> None:

    print("\n")
    print("=" * 60)
    print("SITEPULSE DATA GENERATION SUMMARY")
    print("=" * 60)

    print(
        f"Projects: {len(projects_df):,}"
    )

    print(
        f"Equipment units: {len(equipment_df):,}"
    )

    print(
        f"Daily operation records: "
        f"{len(operations_df):,}"
    )

    print(
        f"Date range: "
        f"{operations_df['date'].min()} "
        f"→ "
        f"{operations_df['date'].max()}"
    )

    print(
        f"Duplicate rows: "
        f"{operations_df.duplicated().sum():,}"
    )

    print(
        "\nMissing values:"
    )

    print(
        operations_df.isna().sum()
    )

    print(
        "\nFuel anomalies:"
    )

    print(
        operations_df[
            "fuel_anomaly_flag"
        ].value_counts()
    )

    print(
        "\nMaterial shortages:"
    )

    print(
        operations_df[
            "material_shortage_flag"
        ].value_counts()
    )

    print(
        "\nEquipment breakdowns:"
    )

    print(
        operations_df[
            "equipment_breakdown_flag"
        ].value_counts()
    )

    print("=" * 60)


# =========================================================
# SAVE DATASETS
# =========================================================

def save_datasets(
    projects_df: pd.DataFrame,
    equipment_df: pd.DataFrame,
    operations_df: pd.DataFrame,
) -> None:

    projects_df.to_csv(
        RAW_DATA_DIR / "projects.csv",
        index=False,
    )

    equipment_df.to_csv(
        RAW_DATA_DIR / "equipment.csv",
        index=False,
    )

    operations_df.to_csv(
        RAW_DATA_DIR
        / "daily_operations.csv",
        index=False,
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main() -> None:

    projects_df = (
        create_projects_dataframe()
    )

    equipment_df = (
        create_equipment_dataframe(
            projects_df
        )
    )

    operations_df = (
        create_daily_operations_dataframe(
            projects_df,
            equipment_df,
        )
    )

    operations_df = (
        inject_data_quality_issues(
            operations_df
        )
    )

    save_datasets(
        projects_df,
        equipment_df,
        operations_df,
    )

    print_validation_summary(
        projects_df,
        equipment_df,
        operations_df,
    )


if __name__ == "__main__":
    main()