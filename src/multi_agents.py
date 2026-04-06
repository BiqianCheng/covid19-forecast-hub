#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CSV Analyzer — Aggregate statistics by state and time window
# Example usage for python3:
#   python3 src/csv_analyzer.py --csv data.csv --state AK --start-date 2024-11-09 --end-date 2024-12-28
#   python3 src/csv_analyzer.py --csv data.csv --sort-by sum --top 10

import argparse
import sys, os
from typing import Optional, List
import pandas as pd
import math
from pathlib import Path
import re
from io import StringIO

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

def Metrics_Calculate(p_args, truth_df, preds):
    if 'target_end_date' in preds.columns:
        preds = preds[preds['target_end_date'] == p_args.targetEndDate]
    preds = preds.rename(columns={'prediction': 'value', 'date': 'target_end_date'})
    # computing the metrics
    median_pred = preds[preds["quantile"] == 0.5]["value"].values[0]
    q10 = preds[preds["quantile"] == 0.1]["value"].values[0]
    q90 = preds[preds["quantile"] == 0.9]["value"].values[0]

    quantiles = preds["quantile"].values
    truth_row = truth_df[truth_df['location'] == str(p_args.state)]
    truth_row = truth_row[truth_row['date'] == p_args.targetEndDate]
    truth_val = truth_row["observation"].values

    # Compute MAE, coverage(q10-q90), and simple WIS
    mae = abs(median_pred - truth_val)
    coverage = int(q10 <= truth_val <= q90)
    
    pred_values = preds["value"].values
    wis_components = []
    for q, pred in zip(quantiles, pred_values):
        if truth_val < pred:
            score = 2 * (1 - q) * (pred - truth_val)
        else:
            score = 2 * q * (truth_val - pred)
        wis_components.append(score)
    simple_wis = sum(wis_components) / len(wis_components)
    return mae[0], simple_wis[0]

def csv_extractor(raw_response):
    # Step 1. Extract the CSV code block (between ```csv and ```)
    match = re.search(r"```csv\n(.*?)```", raw_response, re.DOTALL)
    if not match:
        raise ValueError("No CSV block found in the response!")
    csv_content = match.group(1).strip()
    # Step 2. Read the CSV content into a pandas DataFrame
    df = pd.read_csv(StringIO(csv_content))
    return df


if __name__ == "__main__":
    # pre-define the parameters for dataset filtering
    p = argparse.ArgumentParser(
        description="Read a CSV file, filter rows, and compute summary statistics (sum/mean/max/min)."
    )
    p.add_argument("--truthPath", help="Path to the ground truth file")
    p.add_argument("--state", help="Filter by state code(s), e.g., '06' for CA, '02' for AK")
    p.add_argument("--observeDate", help="Observing date (inclusive), format: YYYY-MM-DD")
    p.add_argument("--targetStartDate", help="Observing date (inclusive), format: YYYY-MM-DD")
    p.add_argument("--targetEndDate", help="Target forecast date (inclusive), format: YYYY-MM-DD")
    p_args = p.parse_args()
    max_iters = 10

    # check if the state code is valid
    if p_args.state not in state_fips_map:
        print(f"Error: Invalid state code '{p_args.state}'. Please provide a valid FIPS state code.")
        sys.exit(1)
    # cut the original table to target dataset
    import pandas as pd
    import datetime as dt
    df = pd.read_parquet(p_args.truthPath, engine='pyarrow')
    df_wk_hosp = df[df['target'] == 'wk inc covid hosp']
    df_wk_hosp['date'] = df_wk_hosp['date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
    df_wk_hosp['as_of'] = df_wk_hosp['as_of'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
    df_wk_hosp = df_wk_hosp[df_wk_hosp['as_of']=='2025-09-24']
    df_wk_hosp = df_wk_hosp[df_wk_hosp['location'] == p_args.state]

    # load the true data for Metrics calculation
    truth_df = pd.read_parquet('target-data/time-series.parquet', engine='pyarrow')
    truth_df = truth_df[truth_df['target'] == 'wk inc covid hosp']
    truth_df['date'] = truth_df['date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
    truth_df['as_of'] = truth_df['as_of'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
    truth_df = truth_df[truth_df['as_of']=='2025-09-24']
    
    # Apply the Gemini API to generate the target prompt
    import google.generativeai as genai
    from pathlib import Path
    genai.configure(api_key="")
    model = genai.GenerativeModel('gemini-2.5-pro')
    # AIzaSyCjRmzYq_84r2YohT1iztXOjpYHEfV9kvQ

    ''' Textualize the original data '''
    # create the Textualize ChatAgent
    chat_txt = model.start_chat(history=[])
    # textualize the Spatial information using Gemini
    state_observation = df_wk_hosp[df_wk_hosp['date']==p_args.observeDate]['observation'].values[0]
    spatial_info = "Target state: " + state_fips_map[p_args.state] + ", total number of hospitalizations=" + str(state_observation)
    spatial_prompt = Path('src/prompts/spatial_prompt.txt').read_text()
    spatial_prompt = spatial_prompt.replace("{spatial_info}", spatial_info)
    spatial_textual = chat_txt.send_message(spatial_prompt).text

    # textualize the Time information using Gemini
    hosp_his = {}
    for row in df_wk_hosp.iterrows():
        hosp_his[row[1]['date']] = row[1]['observation']
        if row[1]['date']== p_args.observeDate:
            break
    hosp_his = str(dict(list(hosp_his.items())[-7:]))
    time_prompt = Path('src/prompts/time_prompt.txt').read_text()
    time_prompt = time_prompt.replace("{time_info}", hosp_his)
    time_prompt = time_prompt.replace("{target_state}", str(state_observation))
    time_textual = chat_txt.send_message(time_prompt).text

    ''' start the Multi-Agent CoT part '''
    # create the Textualize ChatAgent
    chat_pred = model.start_chat(history=[])
    chat_check = model.start_chat(history=[])
    chat_check_flg = False
    MOE_list, simple_wis_list = [], []
    for iters in range(max_iters): 
        # embed the textual data into the system prompt
        sys_prompt = Path('src/prompts/CoT_prompt.txt').read_text()
        sys_prompt = sys_prompt.replace("{spatial_textual}", spatial_textual)
        sys_prompt = sys_prompt.replace("{time_textual}", time_textual)
        sys_prompt = sys_prompt.replace("{target_start_period}", p_args.targetStartDate)
        sys_prompt = sys_prompt.replace("{target_end_period}", p_args.targetEndDate)
        # read the checkAI feedback if exists
        if chat_check_flg:
            sys_prompt = sys_prompt.replace("{checkAI_feedback}", check_response)
        else:
            sys_prompt = sys_prompt.replace("{checkAI_feedback}", "No additional comments.")
        if iters > 5:
            sys_prompt += " . MUST keep the CSV format"
        chat_pred = model.start_chat(history=[])
        # pass the system prompt to Gemini
        response = chat_pred.send_message(sys_prompt).text
        print("Raw response:")
        print(response)
        preds = csv_extractor(response)
        print("CSV extraction:")
        print(preds)
        MOE_i, simple_wis_i = Metrics_Calculate(p_args, truth_df, preds)
        # record the metrics of each iteration - MOE and simple_WIS
        MOE_list.append(MOE_i)
        simple_wis_list.append(simple_wis_i)

        ''' The Check Agent part '''
        # load in the original csv file - the CoT predictions
        if 'target_end_date' in preds.columns:
            preds = preds[preds['target_end_date'] == p_args.targetEndDate]
        preds = preds.rename(columns={'prediction': 'value', 'date': 'target_end_date'})

        # embed the textual data into the system prompt
        check_prompt = Path('src/prompts/check_prompt.txt').read_text()
        check_prompt = check_prompt.replace("{hosp_prediction}", preds.to_csv(index=False))
        check_prompt = check_prompt.replace("{target_state}", state_fips_map[p_args.state])
        check_prompt = check_prompt.replace("{target_start_period}", p_args.targetStartDate)
        check_prompt = check_prompt.replace("{target_end_period}", p_args.targetEndDate)

        # pass the check response
        check_response = chat_check.send_message(check_prompt).text
        chat_check_flg = True
    
    # save the records of metrics
    with open(f'src/CoT_results/multi_agent_metrics/MOE_max-iters-{max_iters}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.txt', 'w') as file:
        for item in MOE_list:
            file.write(f"{item}\n")
    with open(f'src/CoT_results/multi_agent_metrics/simple_WIS_max-iters-{max_iters}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.txt', 'w') as file:
        for item in simple_wis_list:
            file.write(f"{item}\n")
