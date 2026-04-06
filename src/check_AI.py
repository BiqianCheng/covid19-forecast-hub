#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CSV Analyzer — Aggregate statistics by state and time window
# Example usage for python3:
#   python3 src/csv_analyzer.py --csv data.csv --state AK --start-date 2024-11-09 --end-date 2024-12-28
#   python3 src/csv_analyzer.py --csv data.csv --sort-by sum --top 10

import argparse
import sys
from typing import Optional, List
import pandas as pd
import math
from pathlib import Path

state_fips_map = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
}


if __name__ == "__main__":
    # pre-define the parameters for dataset filtering
    p = argparse.ArgumentParser(
        description="Read a CSV file, filter rows, and compute summary statistics (sum/mean/max/min)."
    )
    p.add_argument("--state", help="Filter by state code(s), e.g., '06' for CA, '02' for AK")
    p.add_argument("--observeDate", help="Observing date (inclusive), format: YYYY-MM-DD")
    p.add_argument("--targetStartDate", help="Observing date (inclusive), format: YYYY-MM-DD")
    p.add_argument("--targetEndDate", help="Target forecast date (inclusive), format: YYYY-MM-DD")
    p_args = p.parse_args()

    # check if the state code is valid
    if p_args.state not in state_fips_map:
        print(f"Error: Invalid state code '{p_args.state}'. Please provide a valid FIPS state code.")
        sys.exit(1)
    # load in the original csv file
    csv_path = f'src/CoT_csv/CoT_state-{str(p_args.state)}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.csv'
    preds = pd.read_csv(csv_path)
    if 'target_end_date' in preds.columns:
        preds = preds[preds['target_end_date'] == p_args.targetEndDate]
    preds = preds.rename(columns={'prediction': 'value', 'date': 'target_end_date'})

    # Apply the Gemini API to generate the target prompt
    import google.generativeai as genai
    from pathlib import Path
    genai.configure(api_key="AIzaSyCjRmzYq_84r2YohT1iztXOjpYHEfV9kvQ")
    model = genai.GenerativeModel('gemini-2.5-pro')
    chat = model.start_chat(history=[])

    # embed the textual data into the system prompt
    check_prompt = Path('src/prompts/check_prompt.txt').read_text()
    check_prompt = check_prompt.replace("{hosp_prediction}", preds.to_csv(index=False))
    check_prompt = check_prompt.replace("{target_state}", state_fips_map[p_args.state])
    check_prompt = check_prompt.replace("{target_start_period}", p_args.targetStartDate)
    check_prompt = check_prompt.replace("{target_end_period}", p_args.targetEndDate)
    print(check_prompt)

    chat = model.start_chat(history=[])
    # pass the system prompt to Gemini
    response = chat.send_message(check_prompt).text
    result_path = f'Check-CoT_state-{p_args.state}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.txt'
    out_path = Path('src/CoT_results/check_AI/' + result_path)
    out_path.write_text(response)