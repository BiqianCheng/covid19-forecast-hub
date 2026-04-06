#!/bin/bash
for state_code in $(seq -w 1 56)
do
    python src/csv_analyzer.py --truthPath target-data/time-series.parquet --state $state_code --observeDate 2025-08-30 --targetStartDate 2025-09-13 --targetEndDate 2025-09-20
    python src/csv_analyzer.py --truthPath target-data/time-series.parquet --state $state_code --observeDate 2025-08-30 --targetStartDate 2025-09-06 --targetEndDate 2025-09-13
    echo $state_code
done
