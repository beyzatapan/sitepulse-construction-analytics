from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD FEATURE DATA
# =========================================================

def load_features() -> pd.DataFrame:
    """Load the engineered SitePulse dataset."""

    return pd.read_csv(
        PROCESSED_DATA_DIR
        / "operations_features.csv",
        parse_dates=["date"],
    )


# =========================================================
# PROJECT KPI ENGINE
# =========================================================

def calculate_project_kpis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate daily operational records into
    project-level management KPIs.
    """

    rows = []

    grouped_projects = df.groupby(
        [
            "project_id",
            "project_name",
            "project_type",
            "location",
        ]
    )

    for (
        project_id,
        project_name,
        project_type,
        location,
    ), project_data in grouped_projects:

        project_data = (
            project_data
            .sort_values("date")
            .copy()
        )

        latest = project_data.iloc[-1]

        planned_budget = float(
            project_data[
                "planned_budget_usd"
            ].iloc[0]
        )

        total_cost = float(
            project_data[
                "daily_cost_usd"
            ].sum()
        )

        if planned_budget > 0:
            budget_utilization = (
                total_cost
                / planned_budget
                * 100
            )
        else:
            budget_utilization = np.nan

        risk_days = int(
            project_data[
                "schedule_risk_flag"
            ].sum()
        )

        risk_day_pct = (
            project_data[
                "schedule_risk_flag"
            ].mean()
            * 100
        )

        rows.append(
            {
                "project_id":
                    project_id,

                "project_name":
                    project_name,

                "project_type":
                    project_type,

                "location":
                    location,

                "reporting_date":
                    latest["date"],

                "planned_progress_pct":
                    latest[
                        "planned_progress_pct"
                    ],

                "actual_progress_pct":
                    latest[
                        "actual_progress_pct"
                    ],

                "schedule_variance_pct":
                    latest[
                        "schedule_variance_pct"
                    ],

                "current_schedule_status":
                    latest[
                        "schedule_status"
                    ],

                "average_workforce_ratio":
                    project_data[
                        "workforce_ratio"
                    ].mean(),

                "average_fuel_per_equipment_hour":
                    project_data[
                        "fuel_per_equipment_hour"
                    ].mean(),

                "average_material_delivery_ratio":
                    project_data[
                        "material_delivery_ratio"
                    ].mean(),

                "total_delay_hours":
                    project_data[
                        "delay_hours"
                    ].sum(),

                "latest_7d_delay_hours":
                    latest[
                        "rolling_7d_delay"
                    ],

                "total_cost_usd":
                    total_cost,

                "planned_budget_usd":
                    planned_budget,

                "budget_utilization_pct":
                    budget_utilization,

                "material_shortage_days":
                    int(
                        project_data[
                            "material_shortage_flag"
                        ].sum()
                    ),

                "equipment_breakdown_days":
                    int(
                        project_data[
                            "equipment_breakdown_flag"
                        ].sum()
                    ),

                "fuel_anomaly_days":
                    int(
                        project_data[
                            "fuel_anomaly_flag"
                        ].sum()
                    ),

                "risk_days":
                    risk_days,

                "risk_day_pct":
                    risk_day_pct,

                "current_risk_flag":
                    int(
                        latest[
                            "schedule_risk_flag"
                        ]
                    ),
            }
        )

    project_kpis = pd.DataFrame(rows)

    numeric_columns = (
        project_kpis
        .select_dtypes(
            include="number"
        )
        .columns
    )

    project_kpis[
        numeric_columns
    ] = (
        project_kpis[
            numeric_columns
        ]
        .round(2)
    )

    return project_kpis


# =========================================================
# PORTFOLIO KPI ENGINE
# =========================================================

def calculate_portfolio_kpis(
    project_kpis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create executive-level KPIs across
    the entire construction portfolio.
    """

    portfolio = {
        "number_of_projects":
            len(project_kpis),

        "average_schedule_variance_pct":
            project_kpis[
                "schedule_variance_pct"
            ].mean(),

        "average_actual_progress_pct":
            project_kpis[
                "actual_progress_pct"
            ].mean(),

        "total_portfolio_cost_usd":
            project_kpis[
                "total_cost_usd"
            ].sum(),

        "total_planned_budget_usd":
            project_kpis[
                "planned_budget_usd"
            ].sum(),

        "total_delay_hours":
            project_kpis[
                "total_delay_hours"
            ].sum(),

        "average_fuel_efficiency_lph":
            project_kpis[
                "average_fuel_per_equipment_hour"
            ].mean(),

        "average_workforce_ratio":
            project_kpis[
                "average_workforce_ratio"
            ].mean(),

        "projects_currently_at_risk":
            int(
                project_kpis[
                    "current_risk_flag"
                ].sum()
            ),

        "total_material_shortage_days":
            int(
                project_kpis[
                    "material_shortage_days"
                ].sum()
            ),

        "total_equipment_breakdown_days":
            int(
                project_kpis[
                    "equipment_breakdown_days"
                ].sum()
            ),

        "total_fuel_anomaly_days":
            int(
                project_kpis[
                    "fuel_anomaly_days"
                ].sum()
            ),
    }

    portfolio_df = pd.DataFrame(
        [portfolio]
    )

    numeric_columns = (
        portfolio_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    portfolio_df[
        numeric_columns
    ] = (
        portfolio_df[
            numeric_columns
        ]
        .round(2)
    )

    return portfolio_df


# =========================================================
# VALIDATION
# =========================================================

def validate_project_kpis(
    project_kpis: pd.DataFrame,
) -> None:
    """Validate KPI output."""

    if project_kpis.empty:
        raise ValueError(
            "Project KPI table is empty."
        )

    if project_kpis[
        "project_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate project IDs found "
            "in project KPI table."
        )

    if project_kpis.isna().any().any():
        raise ValueError(
            "Missing values detected "
            "in project KPI table."
        )

    if not project_kpis[
        "actual_progress_pct"
    ].between(0, 100).all():
        raise ValueError(
            "Invalid project progress detected."
        )

    if not project_kpis[
        "budget_utilization_pct"
    ].ge(0).all():
        raise ValueError(
            "Budget utilization cannot be negative."
        )

    if not project_kpis[
        "average_workforce_ratio"
    ].gt(0).all():
        raise ValueError(
            "Invalid workforce ratio detected."
        )


# =========================================================
# SAVE KPI TABLES
# =========================================================

def save_kpis(
    project_kpis: pd.DataFrame,
    portfolio_kpis: pd.DataFrame,
) -> None:

    project_path = (
        PROCESSED_DATA_DIR
        / "project_kpis.csv"
    )

    portfolio_path = (
        PROCESSED_DATA_DIR
        / "portfolio_kpis.csv"
    )

    project_kpis.to_csv(
        project_path,
        index=False,
    )

    portfolio_kpis.to_csv(
        portfolio_path,
        index=False,
    )

    print(
        "\nSaved project KPIs to:"
    )

    print(project_path)

    print(
        "\nSaved portfolio KPIs to:"
    )

    print(portfolio_path)


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    features = load_features()

    project_kpis = (
        calculate_project_kpis(
            features
        )
    )

    validate_project_kpis(
        project_kpis
    )

    portfolio_kpis = (
        calculate_portfolio_kpis(
            project_kpis
        )
    )

    save_kpis(
        project_kpis,
        portfolio_kpis,
    )

    print("\n")
    print("=" * 60)
    print("SITEPULSE KPI ENGINE")
    print("=" * 60)

    print(
        f"Projects processed: "
        f"{len(project_kpis)}"
    )

    print(
        f"Portfolio cost: "
        f"${project_kpis['total_cost_usd'].sum():,.2f}"
    )

    print(
        f"Total delay hours: "
        f"{project_kpis['total_delay_hours'].sum():,.2f}"
    )

    print(
        f"Projects currently at risk: "
        f"{project_kpis['current_risk_flag'].sum()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()