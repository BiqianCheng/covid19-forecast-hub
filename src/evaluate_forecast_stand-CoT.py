import pandas as pd
import argparse
import sys
from pathlib import Path
import datetime as dt

state_fips_map = {
    '01': 'AL',
    '02': 'AK',
    '04': 'AZ',
    '05': 'AR',
    '06': 'CA',
    '08': 'CO',
    '09': 'CT',
    '10': 'DE',
    '11': 'DC',
    '12': 'FL',
    '13': 'GA',
    '15': 'HI',
    '16': 'ID',
    '17': 'IL',
    '18': 'IN',
    '19': 'IA',
    '20': 'KS',
    '21': 'KY',
    '22': 'LA',
    '23': 'ME',
    '24': 'MD',
    '25': 'MA',
    '26': 'MI',
    '27': 'MN',
    '28': 'MS',
    '29': 'MO',
    '30': 'MT',
    '31': 'NE',
    '32': 'NV',
    '33': 'NH',
    '34': 'NJ',
    '35': 'NM',
    '36': 'NY',
    '37': 'NC',
    '38': 'ND',
    '39': 'OH',
    '40': 'OK',
    '41': 'OR',
    '42': 'PA',
    '44': 'RI',
    '45': 'SC',
    '46': 'SD',
    '47': 'TN',
    '48': 'TX',
    '49': 'UT',
    '50': 'VT',
    '51': 'VA',
    '53': 'WA',
    '54': 'WV',
    '55': 'WI',
    '56': 'WY'
}

state_populations = {
    "AL": 5157699,   # Alabama
    "AK": 740133,    # Alaska
    "AZ": 7582384,   # Arizona
    "AR": 3088354,   # Arkansas
    "CA": 39431263,  # California
    "CO": 5957493,   # Colorado
    "CT": 3675069,   # Connecticut
    "DE": 1051917,   # Delaware
    "DC": 702250,    # District of Columbia
    "FL": 23372215,  # Florida
    "GA": 11180878,  # Georgia
    "HI": 1446146,   # Hawaii
    "ID": 2001619,   # Idaho
    "IL": 12710158,  # Illinois
    "IN": 6924275,   # Indiana
    "IA": 3241488,   # Iowa
    "KS": 2970606,   # Kansas
    "KY": 4588372,   # Kentucky
    "LA": 4597740,   # Louisiana
    "ME": 1405012,   # Maine
    "MD": 6263220,   # Maryland
    "MA": 7136171,   # Massachusetts
    "MI": 10140459,  # Michigan
    "MN": 5793151,   # Minnesota
    "MS": 2943045,   # Mississippi
    "MO": 6245466,   # Missouri
    "MT": 1137233,   # Montana
    "NE": 2005465,   # Nebraska
    "NV": 3267467,   # Nevada
    "NH": 1409032,   # New Hampshire
    "NJ": 9500851,   # New Jersey
    "NM": 2130256,   # New Mexico
    "NY": 19867248,  # New York
    "NC": 11046024,  # North Carolina
    "ND": 796568,    # North Dakota
    "OH": 11883304,  # Ohio
    "OK": 4095393,   # Oklahoma
    "OR": 4272371,   # Oregon
    "PA": 13078751,  # Pennsylvania
    "RI": 1112308,   # Rhode Island
    "SC": 5478831,   # South Carolina
    "SD": 924669,    # South Dakota
    "TN": 7227750,   # Tennessee
    "TX": 31290831,  # Texas
    "UT": 3503613,   # Utah
    "VT": 648493,    # Vermont
    "VA": 8811195,   # Virginia
    "WA": 7958180,   # Washington
    "WV": 1769979,   # West Virginia
    "WI": 5960975,   # Wisconsin
    "WY": 587618     # Wyoming
}

parser = argparse.ArgumentParser(description='Evaluate forecast')
parser.add_argument('--observeDate', type=str, required=True, help='Observation date (YYYY-MM-DD)')
parser.add_argument("--targetStartDate", help="Observing date (inclusive), format: YYYY-MM-DD")
parser.add_argument("--targetEndDate", help="Target forecast date (inclusive), format: YYYY-MM-DD")
p_args = parser.parse_args()

# load the true data
truth_df = pd.read_parquet('target-data/time-series.parquet', engine='pyarrow')
truth_df = truth_df[truth_df['target'] == 'wk inc covid hosp']
truth_df['date'] = truth_df['date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
truth_df['as_of'] = truth_df['as_of'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
truth_df = truth_df[truth_df['as_of']=='2025-09-24']

# read the csv files for all states - CoT
MAE_list = []
simple_wis_list = []
for i in range(1, 57):
    state_code = f"{i:02d}"
    csv_path = f'src/CoT_csv/CoT_state-{str(state_code)}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.csv'
    # check if the state code NOT existed
    if str(state_code) not in state_fips_map:
        continue
    preds = pd.read_csv(csv_path)
    if 'target_end_date' in preds.columns:
        preds = preds[preds['target_end_date'] == p_args.targetEndDate]
    preds = preds.rename(columns={'prediction': 'value', 'date': 'target_end_date'})
    preds['value'] = preds['value'] / state_populations[state_fips_map[state_code]] * 10000
    
    median_pred = preds[preds["quantile"] == 0.5]["value"].values[0]
    q10 = preds[preds["quantile"] == 0.1]["value"].values[0]
    q90 = preds[preds["quantile"] == 0.9]["value"].values[0]

    truth_row = truth_df[truth_df['location'] == str(state_code)]
    truth_row = truth_row[truth_row['date'] == p_args.targetEndDate]
    truth_val = truth_row["observation"].values / state_populations[state_fips_map[state_code]] * 10000

    # Compute MAE, coverage(q10-q90), and simple WIS
    mae = abs(median_pred - truth_val)
    coverage = int(q10 <= truth_val <= q90)
    
    quantiles = preds["quantile"].values
    pred_values = preds["value"].values
    wis_components = []
    for q, pred in zip(quantiles, pred_values):
        if truth_val < pred:
            score = 2 * (1 - q) * (pred - truth_val)
        else:
            score = 2 * q * (truth_val - pred)
        wis_components.append(score)
    simple_wis = sum(wis_components) / len(wis_components)
    
    MAE_list.append(mae[0])
    simple_wis_list.append(simple_wis[0])

# read the csv files for all states - Stand
stand_MAE_list = []
stand_simple_wis_list = []
for i in range(1, 57):
    state_code = f"{i:02d}"
    csv_path = f'src/stand_csv/stand_state-{str(state_code)}_obs-{p_args.observeDate}_tgst-{p_args.targetStartDate}_tget-{p_args.targetEndDate}.csv'
    # check if the state code NOT existed
    if str(state_code) not in state_fips_map:
        continue
    preds = pd.read_csv(csv_path)
    preds['value'] = preds['value'] / state_populations[state_fips_map[state_code]] * 10000

    if 'target_end_date' in preds.columns:
        preds = preds[preds['target_end_date'] == p_args.targetEndDate]
    preds = preds.rename(columns={'prediction': 'value', 'date': 'target_end_date'})

    median_pred = preds[preds["quantile"] == 0.5]["value"].values[0]
    q10 = preds[preds["quantile"] == 0.1]["value"].values[0]
    q90 = preds[preds["quantile"] == 0.9]["value"].values[0]

    truth_row = truth_df[truth_df['location'] == str(state_code)]
    truth_row = truth_row[truth_row['date'] == p_args.targetEndDate]
    truth_val = truth_row["observation"].values / state_populations[state_fips_map[state_code]] * 10000

    # Compute MAE, coverage(q10-q90), and simple WIS
    mae = abs(median_pred - truth_val)
    coverage = int(q10 <= truth_val <= q90)

    quantiles = preds["quantile"].values
    pred_values = preds["value"].values
    wis_components = []
    for q, pred in zip(quantiles, pred_values):
        if truth_val < pred:
            score = 2 * (1 - q) * (pred - truth_val)
        else:
            score = 2 * q * (truth_val - pred)
        wis_components.append(score)
    simple_wis = sum(wis_components) / len(wis_components)
    
    stand_MAE_list.append(mae[0])
    stand_simple_wis_list.append(simple_wis[0])

# plot different types of metrics - Performance
import matplotlib.pyplot as plt
import numpy as np

state_names = list(state_fips_map.values())
plt.figure(figsize=(14, 6))
plt.plot(state_names, MAE_list, label='CoT_MAE',linestyle='-', marker='o')
plt.plot(state_names, stand_MAE_list, label='stand_MAE',linestyle='-', marker='x')

plt.xlabel('states', fontweight='bold', fontsize=14)
plt.ylabel('Metrics Value', fontweight='bold', fontsize=14)
plt.title(f'MAE All States, Obs:{p_args.observeDate}, Pred:{p_args.targetStartDate} To {p_args.targetEndDate}')

plt.xticks(rotation=90, fontsize=12) # show x lables 90 degrees angle
plt.tight_layout()

plt.legend()
plt.grid(True)
plt.savefig(f'src/experiment_images/Normal_MAE_AllStates_CoT-VS-Stand_obs-{p_args.observeDate}_tgt-{p_args.targetStartDate}-{p_args.targetEndDate}.png')

plt.figure(figsize=(14, 6))
plt.plot(state_names, simple_wis_list, label='CoT_simple-WIS',linestyle='--', marker='o')
plt.plot(state_names, stand_simple_wis_list, label='stand_simple-WIS',linestyle='--', marker='x')
plt.xlabel('states', fontweight='bold', fontsize=14)
plt.ylabel('Metrics Value', fontweight='bold', fontsize=14)
plt.title(f'simple-WIS All States, Obs:{p_args.observeDate}, Pred:{p_args.targetStartDate} To {p_args.targetEndDate}')

plt.xticks(rotation=90, fontsize=12) # show x lables 90 degrees angle
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig(f'src/experiment_images/Normal_Simple-WIS_AllStates_CoT-VS-Stand_obs-{p_args.observeDate}_tgt-{p_args.targetStartDate}-{p_args.targetEndDate}.png')
plt.show()