import pandas as pd

def evaluate_forecast(forecast_files: str, ground_truth_file: str, locations: list[str], target_date: str):
    
    # reading predictions and ground truth data
    forecast_df = pd.read_csv(forecast_files)
    truth_df = pd.read_csv(ground_truth_file)

    # Predicted value with the given location and target date
    if isinstance(forecast_df['location'][0], str):
        pred = forecast_df[forecast_df['location'] == locations[0]]
        pred = pred[pred['target_end_date'] == target_date]
    else:
        pred = forecast_df[forecast_df['location'] == int(locations[0])]
        pred = pred[pred['target_end_date'] == target_date]
    if pred.empty:
        return f"No prediction data for location={locations}"

    preds = pred.rename(columns={"output_type_id": "quantile"})
    median_pred = preds[preds["quantile"] == 0.5]["value"].values[0]

    q10 = preds[preds["quantile"] == 0.1]["value"].values[0]
    q90 = preds[preds["quantile"] == 0.9]["value"].values[0]

    truth_row = truth_df[truth_df['location'] == locations[0]]
    truth_row = truth_row[truth_row['date'] == target_date]
    if truth_row.empty:
        return f"No ground truth for location={location}"

    truth_val = truth_row["value"].values[0]

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
    
    # output the evaluation metrics
    return {
        "location (FIPS)": locations[0],
        "truth_value": truth_val,
        "median_prediction": round(median_pred, 2),
        "MAE": round(mae, 2),
        "q10": round(q10, 2),
        "q90": round(q90, 2),
        "coverage_10_90": coverage,
        "simple_WIS": round(simple_wis, 2)
    }

# run evaluation example
if __name__ == "__main__":
    ### Compute the merics of different models
    model_list = ["output", "output_CoT", "UMass-ar6_pooled", "UM-DeepOutbreak", "UMass-gbqr", "Google_SAI-Ensemble", "NEU_ISI-AdaptiveEnsemble"]
    ground_truth_file = "target-data/covid-hospital-admissions.csv"
    locations = ["06"] # California
    start_date = "2025-03-22"
    target_date = "2025-03-29"
    # compute the matrics for each model VS truth data
    results_list = {}
    for model in model_list:
        if model == "output" or model == "output_CoT":
            forecast_file = "src/" + model + ".csv"
        else:
            forecast_file = "model-output/" + model + "/" + start_date + "-" + model + ".csv"
        results = evaluate_forecast(forecast_file, ground_truth_file, locations, target_date)
        results_list[model] = results
    
    ### Compute the merics of different models
    model_list_AK = ["output_AK", "output_AK_CoT", "UMass-ar6_pooled", "UM-DeepOutbreak", "UMass-gbqr", "Google_SAI-Ensemble", "NEU_ISI-AdaptiveEnsemble"]
    ground_truth_file = "target-data/covid-hospital-admissions.csv"
    locations = ["02"] # Alaska
    start_date = "2025-03-22"
    target_date = "2025-03-29"
    # compute the matrics for each model VS truth data
    results_list_AK = {}
    for model in model_list_AK:
        if model == "output_AK" or model == "output_AK_CoT":
            forecast_file = "src/" + model + ".csv"
        else:
            forecast_file = "model-output/" + model + "/" + start_date + "-" + model + ".csv"
        results = evaluate_forecast(forecast_file, ground_truth_file, locations, target_date)
        results_list_AK[model] = results
    
    ### Compute the merics of different models
    model_list_AL = ["output_AL", "output_AL_CoT", "UMass-ar6_pooled", "UM-DeepOutbreak", "UMass-gbqr", "Google_SAI-Ensemble", "NEU_ISI-AdaptiveEnsemble"]
    ground_truth_file = "target-data/covid-hospital-admissions.csv"
    locations = ["01"] # Alaska
    start_date = "2025-03-22"
    target_date = "2025-03-29"
    # compute the matrics for each model VS truth data
    results_list_AL = {}
    for model in model_list_AL:
        if model == "output_AL" or model == "output_AL_CoT":
            forecast_file = "src/" + model + ".csv"
        else:
            forecast_file = "model-output/" + model + "/" + start_date + "-" + model + ".csv"
        results = evaluate_forecast(forecast_file, ground_truth_file, locations, target_date)
        results_list_AL[model] = results
    
    # plot different types of metrics - Performance
    import matplotlib.pyplot as plt
    import numpy as np
    print(results_list['output']['MAE'])

    MAE_list_CA = [results_list[model]['MAE'] for model in model_list]
    WIS_list_CA = [results_list[model]['simple_WIS'] for model in model_list]
    MAE_list_AK = [results_list_AK[model]['MAE'] for model in model_list_AK]
    WIS_list_AK = [results_list_AK[model]['simple_WIS'] for model in model_list_AK]
    MAE_list_AL = [results_list_AL[model]['MAE'] for model in model_list_AL]
    WIS_list_AL = [results_list_AL[model]['simple_WIS'] for model in model_list_AL]
    model_names = ["our_single", "our_CoT", "UMass-ar6_pooled", "UM-DeepOutbreak", "UMass-gbqr", "Google_SAI", "NEU_ISI"]
    plt.figure(1)
    plt.plot(model_names, MAE_list_CA, label='MAE-CA',linestyle='-', marker='o')
    plt.plot(model_names, WIS_list_CA, label='simple-WIS-CA',linestyle='--', marker='x')
    plt.plot(model_names, MAE_list_AK, label='MAE-AK',linestyle='-', marker='o')
    plt.plot(model_names, WIS_list_AK, label='simple-WIS-AK',linestyle='--', marker='x')
    plt.plot(model_names, MAE_list_AL, label='MAE-AL',linestyle='-', marker='o')
    plt.plot(model_names, WIS_list_AK, label='simple-WIS-AL',linestyle='--', marker='x')

    plt.xlabel('model names', fontweight='bold', fontsize=14)
    plt.ylabel('Metrics Value', fontweight='bold', fontsize=14)
    plt.title('Comparison of MAE and simple-WIS for Different Models - Multiple States')
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
    # ### Compute the merics of consecutive 10 experiments of CoT model
    # model_list = ["output"+str(i+1) for i in range(10)]
    # ground_truth_file = "target-data/covid-hospital-ad_ALmissions.csv"
    # locations = ["02"] # California
    # start_date = "2025-03-22"
    # target_date = "2025-03-29"
    # # compute the matrics for each model VS truth data
    # results_list = {}
    # for model in model_list:
    #     forecast_file = "src/outputs_CoT/" + model + ".csv"
    #     results = evaluate_forecast(forecast_file, ground_truth_file, locations, target_date)
    #     results_list[model] = results
    # # plot metrics of 10 experiments - Stability
    # # plot different types of metrics - Performance
    # import matplotlib.pyplot as plt
    # import numpy as np

    # MAE_list = [results_list[model]["MAE"] for model in model_list]
    # WIS_list = [results_list[model]["simple_WIS"] for model in model_list]
    # model_names = ["CoT Exp."+str(i+1) for i in range(10)]
    # plt.figure(2)
    # plt.plot(model_names, MAE_list, label='MAE', color='blue', linestyle='-', marker='o')
    # plt.plot(model_names, WIS_list, label='simple-WIS', color='red', linestyle='--', marker='x')

    # plt.xlabel('Experiment Trials', fontweight='bold', fontsize=14)
    # plt.ylabel('Metrics Value', fontweight='bold', fontsize=14)
    # plt.title('CoT Stability Test by multiple experiments')
    # plt.legend()
    # plt.grid(True)
    
    # plt.show()