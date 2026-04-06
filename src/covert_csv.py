import re
import pandas as pd
from io import StringIO

import argparse
import sys
from pathlib import Path

p = argparse.ArgumentParser(description="Specify the parameters for extracting CSV from text.")
p.add_argument("--state", type=str, help="The state code")
p.add_argument("--observeDate", type=str, help="Observing date (inclusive), format: YYYY-MM-DD")
p.add_argument("--targetStartDate", help="Observing date (inclusive), format: YYYY-MM-DD")
p.add_argument("--targetEndDate", help="Target forecast date (inclusive), format: YYYY-MM-DD")
p_args = p.parse_args()

# Load your text (in practice, read from a .txt file)
result_path = f'stand_state-{p_args.state}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.txt'
file_path = Path('src/stand_results/' + result_path)
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Step 1. Extract the CSV code block (between ```csv and ```)
match = re.search(r"```csv\n(.*?)```", text, re.DOTALL)
if not match:
    raise ValueError("No CSV block found in the text!")
csv_content = match.group(1).strip()

# Step 2. Read the CSV content into a pandas DataFrame
df = pd.read_csv(StringIO(csv_content))
print(df)

# Optionally, save it to a CSV file
output_path = f'stand_state-{p_args.state}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.csv'
output_path = 'src/stand_csv/' + output_path
df.to_csv(output_path, index=False)
print("Succefully saved the CSV to:", output_path, "! Please relocate to fine the files.")
