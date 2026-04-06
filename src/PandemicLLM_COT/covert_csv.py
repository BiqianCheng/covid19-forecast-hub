import pandas as pd
import io
import re

# Step 1: Read the .txt file content
with open("src/pred_quantiles.txt", "r") as file:
    raw_text = file.read()

data_io = io.StringIO(raw_text.strip())
df = pd.read_csv(data_io)

# Step 5: Save as .csv
df.to_csv("src/output.csv", index=False)
print("Conversion complete! Saved as 'output.csv'.")