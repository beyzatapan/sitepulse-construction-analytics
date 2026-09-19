# SitePulse Dataset

The SitePulse dataset is a synthetic construction operations dataset generated specifically for this project.

No confidential or real company data is included.

## Dataset Files

The generator creates three datasets inside `data/raw/`.

### projects.csv

Contains project-level information.

Main fields:

- project_id
- project_name
- project_type
- location
- planned_budget_usd
- planned_duration_days
- start_date
- planned_end_date

### equipment.csv

Contains equipment assigned to construction projects.

Main fields:

- equipment_id
- project_id
- equipment_type
- fuel_type
- expected_fuel_per_hour

### daily_operations.csv

Contains daily operational observations for each project.

Main fields:

- date
- project_id
- workers
- equipment_hours
- fuel_consumption_l
- material_delivered_t
- material_used_t
- planned_progress_pct
- actual_progress_pct
- daily_cost_usd
- weather
- delay_hours
- material_shortage_flag
- equipment_breakdown_flag
- fuel_anomaly_flag

## Data Generation

The datasets are generated using:

```bash
python src/sitepulse/generate_data.py

A fixed random seed is used to make the generated data reproducible.

Simulated Operational Relationships

The generator includes relationships between:

* weather and productivity
* workforce levels and construction progress
* equipment usage and fuel consumption
* material shortages and delays
* equipment breakdowns and delays
* delays and operational costs

Controlled fuel anomalies are also introduced for later anomaly detection experiments.

Data Quality Issues

To simulate real-world operational data, the generator intentionally introduces:

* missing values
* duplicate records
* fuel consumption anomalies
* material shortages
* equipment breakdown events

These issues are intentionally preserved in the raw dataset and will be handled later in the data-cleaning pipeline.

Privacy

All project names, values, operational events, and records are synthetic.

No proprietary construction company information is stored in this repository

Sonra kaydetmek için:

```text

