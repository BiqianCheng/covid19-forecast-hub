#!/bin/bash
for state_code in $(seq -w 1 56)
do
    python src/check_AI.py --state $state_code --observeDate 2025-08-30 --targetStartDate 2025-09-13 --targetEndDate 2025-09-20
    python src/check_AI.py --state $state_code --observeDate 2025-08-30 --targetStartDate 2025-09-06 --targetEndDate 2025-09-13
    echo $state_code
done
