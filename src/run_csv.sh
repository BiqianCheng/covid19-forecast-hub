#!/bin/bash
for state_code in $(seq -w 1 56)
do
    python src/covert_csv.py --state $state_code --observeDate 2025-08-30 --targetStartDate 2025-09-13 --targetEndDate 2025-09-20
    echo $state_code
done
