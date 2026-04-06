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

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read a CSV file, filter rows, and compute summary statistics (sum/mean/max/min)."
    )
    p.add_argument("--csv", required=True, help="Path to the CSV file")
    p.add_argument("--state", help="Filter by state code(s), e.g., AK or multiple like AK,CA,NY")
    p.add_argument("--start-date", help="Start date (inclusive), format: YYYY-MM-DD")
    p.add_argument("--end-date", help="End date (inclusive), format: YYYY-MM-DD")
    p.add_argument("--state-col", default="state", help="Column name for state (default: state)")
    p.add_argument("--date-col", default="date", help="Column name for date (default: date)")
    p.add_argument("--value-col", default="value", help="Column name for numeric values (default: value)")
    p.add_argument("--date-format", default=None, help="Optional date format, e.g., %Y-%m-%d")
    p.add_argument("--sort-by", choices=["sum", "mean", "max", "min", "count"],
                   help="Aggregate by state and sort by this metric (ignores --state)")
    p.add_argument("--desc", action="store_true", help="Sort in descending order (default: ascending)")
    p.add_argument("--top", type=int, default=None, help="Show only the top N results (with --sort-by)")
    p.add_argument("--dropna", action="store_true",
                   help="Drop rows with missing or non-numeric values in the value column")
    return p.parse_args(argv)


def read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def coerce_types(df: pd.DataFrame, date_col: str, value_col: str, date_format: Optional[str] = None) -> pd.DataFrame:
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df


def apply_filters(
    df: pd.DataFrame,
    state_col: str,
    states: Optional[List[str]],
    date_col: str,
    start_date: Optional[str],
    end_date: Optional[str],
    dropna_value: bool,
    value_col: str,
) -> pd.DataFrame:
    out = df.copy()
    if states:
        states_norm = [s.strip() for s in states if s.strip()]
        out = out[out[state_col].astype(str).isin(states_norm)]
    if start_date:
        out = out[out[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out[date_col] <= pd.to_datetime(end_date)]
    if dropna_value:
        out = out[~out[value_col].isna()]
    return out


def summarize_for_states(df: pd.DataFrame, state_col: str, value_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(state_col, dropna=False)[value_col]
        .agg(count="count", sum="sum", mean="mean", max="max", min="min")
        .reset_index()
    )
    grouped["mean"] = grouped["mean"].astype(float).round(3)
    return grouped


def print_state_summary(df: pd.DataFrame, state: str, state_col: str, value_col: str,
                        start: Optional[str], end: Optional[str]):
    sub = df[df[state_col] == state]
    if sub.empty:
        print(f"⚠️ No records found for state {state} in the given filters.")
        return
    # do not consider the true value of target date
    total = sub[value_col].sum() - sub[value_col].iloc[-1]
    count = sub[value_col].count()
    mean = math.floor(sub[value_col].mean())
    vmax = sub[value_col].max()
    vmin = sub[value_col].min()

    print(f"=== Summary for state {state} ({start or 'earliest'} ~ {end or 'latest'}, {count} records) ===")
    print(f"Sum: {total}")
    # print(f"Mean: {mean:.3f}" if pd.notna(mean) else "Mean: NA")
    print(f"Mean: {mean}" if pd.notna(mean) else "Mean: NA")
    print(f"Max: {vmax}" if pd.notna(vmax) else "Max: NA")
    print(f"Min: {vmin}" if pd.notna(vmin) else "Min: NA")
    print("")
    return {"state": state, "info": {"sum": total, "mean": mean, "max": vmax, "min": vmin}, \
            "time": {"start_time": start, "end_time": end}}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    df = read_csv(args.csv)
    required_cols = [args.state_col, args.date_col, args.value_col]
    for c in required_cols:
        if c not in df.columns:
            sys.stderr.write(f"❌ Missing required column: {c}; current columns: {list(df.columns)}\n")
            return 2

    df = coerce_types(df, args.date_col, args.value_col, args.date_format)
    states = args.state.split(",") if args.state else None
    filtered = apply_filters(
        df,
        state_col=args.state_col,
        states=states,
        date_col=args.date_col,
        start_date=args.start_date,
        end_date=args.end_date,
        dropna_value=args.dropna,
        value_col=args.value_col,
    )

    if args.sort_by:
        summary = summarize_for_states(filtered, args.state_col, args.value_col)
        summary = summary.sort_values(by=args.sort_by, ascending=not args.desc, na_position="last")
        if args.top:
            summary = summary.head(args.top)

        print("=== Ranking by state ===")
        print(f"Date window: {args.start_date or 'earliest'} ~ {args.end_date or 'latest'}; "
              f"Metric: {args.sort_by}; Order: {'descending' if args.desc else 'ascending'}")
        for _, row in summary.iterrows():
            print(
                f"{row[args.state_col]} -> count={int(row['count'])}, sum={row['sum']}, "
                f"mean={row['mean']}, max={row['max']}, min={row['min']}"
            )
        return 0

    if states:
        output = {}
        for s in states:
            s_stats = print_state_summary(filtered, s.strip(), args.state_col, args.value_col,
                                args.start_date, args.end_date)
            output = {**output, **s_stats}
        return output

    summary = summarize_for_states(filtered, args.state_col, args.value_col)
    print("No --state provided; showing summary for all states:")
    print(f"Date window: {args.start_date or 'earliest'} ~ {args.end_date or 'latest'}")
    for _, row in summary.iterrows():
        print(
            f"{row[args.state_col]} -> count={int(row['count'])}, sum={row['sum']}, "
            f"mean={row['mean']}, max={row['max']}, min={row['min']}"
        )
    return 0

if __name__ == "__main__":
    # raise SystemExit(main())
    output = main()

    # read extra background if provided
    from pathlib import Path
    extra_background_path = Path('src/extra_background.txt')
    if extra_background_path.exists():
        extra_background = extra_background_path.read_text()
    else:
        extra_background = ""
    output["extra_background"] = extra_background  # save into result
    print("=== Extra Background Knowledge ===")
    print(extra_background)
    
    # ask the user which option to use for textualization
    print("Choose the option for textualization:")
    print("1. Use hardcoded prompt template (simple and fast)")
    print("2. Use Gemini API to generate prompt (more accurate but slower)")
    print("3. Apply long history data and Gemini API")
    choice = input("Enter one of [1, 2, 3]: ").strip()
    if choice == '1':
        spatial_textual = "The total number of hospitalizations in " + output["state"] + " is " + str(output["info"]["sum"])
        time_textual = "During peroid: from " + str(output["time"]["start_time"]) + \
                        " to " + str(output["time"]["end_time"]) + ":\n" + \
                        "The statistics of hospitalizations change are: " + "Mean=" + str(output["info"]["mean"]) + \
                        ", Max=" + str(output["info"]["max"]) + ", Min=" + str(output["info"]["min"]) + "."
    elif choice == '2' or choice == '3':
        # using Gemini API to textualize multi-modal information
        # import google.generativeai as genai
        # from pathlib import Path
        # genai.configure(api_key="AIzaSyByh6W10hlX1KB_-AlvBDlAVJSFdZR6gHU")
        # model = genai.GenerativeModel('gemini-2.5-pro')
        # chat = model.start_chat(history=[])
        
        # textualize the spatial information
        spatial_info = "Target state: " + output["state"] + ", total number of hospitalizations=" + str(output["info"]["sum"])
        spatial_prompt = Path('src/prompts/spatial_prompt.txt').read_text()
        spatial_prompt = spatial_prompt.replace("{spatial_info}", spatial_info)
        print("# Spatia prompt:")
        print(spatial_prompt)
        # spatial_textual = chat.send_message(spatial_prompt).text
        # textualize the time related information
        if choice == '2':
            time_info = "Target state: " + output["state"] + \
                        ", Target time period: from " + str(output["time"]["start_time"]) + \
                        " to " + str(output["time"]["end_time"]) + ":\n" + \
                        "The statistics data of hospitalizations trend are: " + "Mean=" + str(output["info"]["mean"]) + \
                        ", Max=" + str(output["info"]["max"]) + ", Min=" + str(output["info"]["min"]) + "."
            time_prompt = Path('src/prompts/time_prompt.txt').read_text()
            time_prompt = time_prompt.replace("{time_info}", time_info)
            # time_textual = chat.send_message(time_prompt).text
        elif choice == '3':
            args = parse_args()
            df = read_csv(args.csv)
            # select the target data
            filtered = df[df['state'] == args.state]
            current_idx = filtered[filtered['date'] == args.start_date].index[0]
            filtered = filtered.loc[:current_idx]
            # the history of data trends
            hosp_his = {}
            for row in filtered.iterrows():
                hosp_his[row[1]['date']] = row[1]['value']
            hosp_his = str(hosp_his)
            print(hosp_his)
            # send the time trends to Gemini
            time_prompt = Path('src/prompts/time_prompt.txt').read_text()
            time_prompt = time_prompt.replace("{time_info}", hosp_his)
            time_prompt = time_prompt.replace("{target_state}", output["state"])
            print("# Time prompt:")
            print(time_prompt)
            # time_textual = chat.send_message(time_prompt).text
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
        
    # embed the textual data into the system prompt
    sys_prompt = Path('src/prompts/prompt.txt').read_text()
    sys_prompt = sys_prompt.replace("{spatial_textual}", spatial_textual)
    sys_prompt = sys_prompt.replace("{time_textual}", time_textual)
    sys_prompt = sys_prompt.replace("{extra_background}", extra_background)
    sys_prompt = sys_prompt.replace("{target_start_period}", str(output["time"]["start_time"]))
    sys_prompt = sys_prompt.replace("{target_end_period}", str(output["time"]["end_time"]))

    Path('src/sys_prompt.txt').write_text(sys_prompt)

    out_path = Path('src/sys_prompt.txt')
    out_path.write_text(sys_prompt)

    print(f"System prompt has been saved to {out_path}")
    print("You can open it with:  cat src/sys_prompt.txt")