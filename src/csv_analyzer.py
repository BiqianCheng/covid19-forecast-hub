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
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
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
    total = sub[value_col].sum()
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
    spatial_textual = "The total number of hospitalizations in " + output["state"] + " is " + str(output["info"]["sum"])
    time_textual = "During peroid: from " + str(output["time"]["start_time"]) + \
                    " to " + str(output["time"]["end_time"]) + ":\n" + \
                    "The statistics of hospitalizations change are: " + "Mean=" + str(output["info"]["mean"]) + \
                    ", Max=" + str(output["info"]["max"]) + ", Min=" + str(output["info"]["min"]) + "."
    sys_prompt = Path('src/prompt.txt').read_text()
    sys_prompt = sys_prompt.replace("{spatial_textual}", spatial_textual)
    sys_prompt = sys_prompt.replace("{time_textual}", time_textual)
    Path('src/sys_prompt.txt').write_text(sys_prompt)