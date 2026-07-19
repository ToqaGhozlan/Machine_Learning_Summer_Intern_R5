import pandas as pd
df = pd.read_csv("final_internship_data.csv")
print("Weather unique values:", df["Weather"].unique())
print("Car Condition unique values:", df["Car Condition"].unique())
print("Traffic Condition unique values:", df["Traffic Condition"].unique())