import json
import matplotlib.pyplot as plt
import numpy as np

with open("evaluations/baseline_results_with_asr.json", "r") as f:
    baseline_results = json.load(f)

with open("evaluations/defense_results.json", "r") as f:
    defense_results = json.load(f)

def compare_mixed_results(baseline_results, defense_results):
    models = list(baseline_results.keys())

    baseline_f1_scores = []
    defense_f1_scores = []
    baseline_asr_scores = []
    defense_asr_scores = []

    for model in models:

        if "mixed" in baseline_results[model]:
            baseline_f1_scores.append(baseline_results[model]["mixed"]["F1 Score"])
            baseline_asr_scores.append(baseline_results[model]["mixed"]["ASR"])
        else:
            print(f"Model {model} does not have 'mixed' results in baseline.")  # Debug print

        if "mixed" in defense_results[model]:
            defense_f1_scores.append(defense_results[model]["mixed"]["F1 Score"])
            defense_asr_scores.append(defense_results[model]["mixed"]["ASR"])
        else:
            print(f"Model {model} does not have 'mixed' results in defense.")  # Debug print

    if not baseline_f1_scores or not defense_f1_scores or not baseline_asr_scores or not defense_asr_scores:
        print("No 'mixed' results found in either baseline or defense results.")
        return

    max_f1_score = max(max(baseline_f1_scores), max(defense_f1_scores))
    min_f1_score = min(min(baseline_f1_scores), min(defense_f1_scores))
    y_label_f1 = f"F1 Score (Range: {min_f1_score:.2f} - {max_f1_score:.2f})"

    max_asr_score = max(max(baseline_asr_scores), max(defense_asr_scores))
    min_asr_score = min(min(baseline_asr_scores), min(defense_asr_scores))
    y_label_asr = f"ASR (Range: {min_asr_score:.2f} - {max_asr_score:.2f})"

     # Create line chart for F1

    plt.figure(figsize=(8, 4))
    x_positions = np.arange(len(models))  # Space out x positions
    plt.subplot(1, 2, 1)
    plt.plot(x_positions, baseline_f1_scores, marker='o', label="Baseline F1 Score", color="gray", linestyle='--')
    plt.plot(x_positions, defense_f1_scores, marker='o', label="Defense F1 Score", color="blue", linestyle='-')

    plt.title("Baseline vs Defense F1 Score", fontsize=14)
    plt.xlabel("Models")
    plt.ylabel(y_label_f1)
    plt.ylim(min_f1_score - 0.05, max_f1_score + 0.05)  # Adjust the y-axis range for better clarity
    plt.xticks(x_positions, models, rotation=30, ha='right', fontsize=10)  # Rotate labels slightly
    plt.legend()

    for i, (b_score, d_score) in enumerate(zip(baseline_f1_scores, defense_f1_scores)):
        plt.text(i, b_score + 0.002, f"{b_score:.3f}", ha="center", va="bottom", fontsize=9, color="gray")
        plt.text(i, d_score - 0.002, f"{d_score:.3f}", ha="center", va="top", fontsize=9, color="blue")

    # Create line chart for ASR
    plt.subplot(1, 2, 2)
    plt.plot(x_positions, baseline_asr_scores, marker='o', label="Baseline ASR", color="gray", linestyle='--')
    plt.plot(x_positions, defense_asr_scores, marker='o', label="Defense ASR", color="blue", linestyle='-')


    plt.title("Baseline vs Defense ASR", fontsize=14)
    plt.xlabel("Models")
    plt.ylabel(y_label_asr)
    plt.ylim(min_asr_score - 0.05, max_asr_score + 0.05)  # Adjust the y-axis range for better clarity
    plt.xticks(x_positions, models, rotation=30, ha='right', fontsize=10)  # Rotate labels slightly
    plt.legend()


    for i, (b_score, d_score) in enumerate(zip(baseline_asr_scores, defense_asr_scores)):
        plt.text(i, b_score + 0.002, f"{b_score:.3f}", ha="center", va="bottom", fontsize=9, color="gray")
        plt.text(i, d_score - 0.002, f"{d_score:.3f}", ha="center", va="top", fontsize=9, color="blue")

    plt.tight_layout()
    plt.savefig("evaluations/baseline_vs_defense_mixed_comparison_adjusted.png")
    plt.show()

# Run the comparison
compare_mixed_results(baseline_results, defense_results)