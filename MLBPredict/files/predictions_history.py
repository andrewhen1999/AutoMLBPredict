import pandas as pd

# Load both CSV files into DataFrames
df1 = pd.read_csv(r"C:\Users\andre\OneDrive\Desktop\MLBAttempt\AutoMLBPredict\MLBPredict\output\predictions_history.csv")
df2 = pd.read_csv(r"C:\Users\andre\OneDrive\Desktop\MLBAttempt\AutoMLBPredict\MLBPredict\output\predictions.csv")

# Combine them vertically
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save back to the destination file
combined_df.to_csv(r"MLBPredict/output/predictions_history.csv" , index=False)
