# plot different types of metrics - Performance
import matplotlib.pyplot as plt
import numpy as np

MAE_file_path = 'src/CoT_results/multi_agent_metrics/MOE_max-iters-5_obs-2025-08-30_tgst-2025-09-06_tget-2025-09-13.txt'
WIS_file_path = 'src/CoT_results/multi_agent_metrics/simple_WIS_max-iters-5_obs-2025-08-30_tgst-2025-09-06_tget-2025-09-13.txt'

with open(MAE_file_path, 'r') as file:
    lines_list = file.readlines()
MAE_float_list = [float(i.strip()) for i in lines_list if i.strip()]
with open(WIS_file_path, 'r') as file:
    lines_list = file.readlines()
WIS_float_list = [float(i.strip()) for i in lines_list if i.strip()]
iter_num = [i for i in range(5)]

plt.figure(figsize=(14, 6))
plt.plot(iter_num, MAE_float_list, label='MAE',linestyle='-', marker='o')
plt.plot(iter_num, WIS_float_list, label='simple_WIS',linestyle='-', marker='x')

plt.xlabel('iters', fontweight='bold', fontsize=14)
plt.ylabel('Metrics Value', fontweight='bold', fontsize=14)
plt.title(f'MAE and simple_WIS for 5 pred - check iters')

plt.xticks(rotation=90, fontsize=12) # show x lables 90 degrees angle
plt.tight_layout()

plt.legend()
plt.grid(True)
plt.savefig(f'src/experiment_images/multi-agent-test_metrics-AK.png')
