import pandas as pd


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/raw/daily_operations.csv")


# =========================================================
# BASIC VALIDATION
# =========================================================

print("\nSITEPULSE DATASET VERIFICATION")
print("=" * 50)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
for column in df.columns:
    print(f"- {column}")

print("\nData types:")
for column, dtype in df.dtypes.items():
    print(f"- {column}: {dtype}")

print("\nMissing values:")
for column, missing_count in df.isna().sum().items():
    if missing_count > 0:
        print(f"- {column}: {missing_count}")

print(f"\nDuplicate rows: {df.duplicated().sum()}")

print("\nProject count:")
print(df["project_id"].nunique())

print("\nDate range:")
print(f"{df['date'].min()} -> {df['date'].max()}")

print("\nFuel anomaly count:")
print(df["fuel_anomaly_flag"].sum())

print("\nMaterial shortage count:")
print(df["material_shortage_flag"].sum())

print("\nEquipment breakdown count:")
print(df["equipment_breakdown_flag"].sum())

print("=" * 50)