import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from file
with open("evaluations/baseline_results_with_asr.json", "r") as f:
    results = json.load(f)

# Average F1 and ASR scores across all models for each perturbation method
def plot_average_f1_and_asr(results):
    models = list(results.keys())
    methods = list(results[models[0]].keys()) 
    methods.remove("original")  # Remove the original method 

    # Calculate average F1 and ASR scores for each method
    average_f1_diffs = []
    average_asr_diffs = []

    for method in methods:
        avg_f1_diff = np.mean([results[model][method]["F1 Score"] - results[model]["original"]["F1 Score"] for model in models])
        avg_asr_diff = np.mean([results[model][method]["ASR"] - results[model]["original"]["ASR"] for model in models])
        average_f1_diffs.append(avg_f1_diff)
        average_asr_diffs.append(avg_asr_diff)

    combined_scores = list(zip(average_f1_diffs, average_asr_diffs, methods))

    # Sort the scores 
    sorted_scores = sorted(combined_scores, reverse=True)
    sorted_f1_diffs, sorted_asr_diffs, sorted_methods = zip(*sorted_scores)

    x = np.arange(len(sorted_methods))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - bar_width / 2, sorted_f1_diffs, bar_width, label="Average F1 Score Difference", color='blue', alpha=0.7)
    bars2 = ax.bar(x + bar_width / 2, sorted_asr_diffs, bar_width, label="Average ASR Difference", color='orange', alpha=0.7)

    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods, rotation=45)
    ax.set_ylabel("Score Differences")
    ax.legend()


    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', va='bottom')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("evaluations/impact_of_methods_comparison.png")
    plt.show()

# Call the function to create the plot
plot_average_f1_and_asr(results)