import pandas as pd
# import json

# # Load your training dataset
df = pd.read_csv(r"C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model\household_power_data_2023_2024.csv")  # Replace with actual file

print(df.columns)
# # Calculate threshold (e.g., 98% of historical max load)
# threshold_value = df['Electric Load (MW)'].max() * 0.98

# # Save to JSON
# with open("threshold.json", "w") as f:
#     json.dump({"threshold": threshold_value}, f)

# print(f"Threshold saved: {threshold_value}")
# import os

# folder_path = r'C:\Users\Dell\Documents\Rajat Nair\SCOPE_OJT\Electricity_Forecasting\pow_app\model'  # replace with your folder path
# file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# print(file_names)
