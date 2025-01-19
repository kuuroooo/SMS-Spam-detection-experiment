import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

with open('evaluations/baseline_results_with_asr.json') as f:
    baseline_results = json.load(f)

with open('evaluations/ensemble_results.json') as f:
    ensemble_results = json.load(f)

baseline_methods = ['Naive Bayes', 'SVM', 'Decision Tree', 'LSTM']
baseline_f1_scores = [baseline_results[method]['mixed']['F1 Score'] for method in baseline_methods]
baseline_asr_scores = [baseline_results[method]['mixed']['ASR'] for method in baseline_methods]
ensemble_methods = ['Voting Ensemble', 'Stacking Ensemble']
ensemble_f1_scores = [ensemble_results[method]['mixed']['F1 Score'] for method in ensemble_methods]
ensemble_asr_scores = [ensemble_results[method]['mixed']['ASR'] for method in ensemble_methods]


methods = baseline_methods + ensemble_methods
f1_scores = baseline_f1_scores + ensemble_f1_scores
asr_scores = baseline_asr_scores + ensemble_asr_scores

# Sort methods and scores by F1 scores (highest to lowest)
sorted_f1_methods_scores = sorted(zip(f1_scores, methods), reverse=True)
sorted_f1_scores, sorted_f1_methods = zip(*sorted_f1_methods_scores)

# Sort methods and scores by ASR scores (lowest to highest)
sorted_asr_methods_scores = sorted(zip(asr_scores, methods))
sorted_asr_scores, sorted_asr_methods = zip(*sorted_asr_methods_scores)

colors_f1 = ['grey' if method in baseline_methods else 'orange' for method in sorted_f1_methods]
colors_asr = ['grey' if method in baseline_methods else 'orange' for method in sorted_asr_methods]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot F1 scores
ax1.plot(range(len(sorted_f1_methods)), sorted_f1_scores, linestyle='--', color='lightgrey', marker='o')
for i, (score, color) in enumerate(zip(sorted_f1_scores, colors_f1)):
    ax1.plot(i, score, marker='o', color=color)
for i, score in enumerate(sorted_f1_scores):
    ax1.text(i, score - 0.005, f'{score:.3f}', ha='center', fontsize=9)
ax1.set_xticks(range(len(sorted_f1_methods)))
ax1.set_xticklabels(sorted_f1_methods, rotation=45, ha='right')
ax1.set_ylabel('F1 Score')
ax1.set_title('Comparison of Baseline and Ensemble Methods (F1 Score)')

# Plot ASR scores
ax2.plot(range(len(sorted_asr_methods)), sorted_asr_scores, linestyle='--', color='lightgrey', marker='o')
for i, (score, color) in enumerate(zip(sorted_asr_scores, colors_asr)):
    ax2.plot(i, score, marker='o', color=color)
for i, score in enumerate(sorted_asr_scores):
    ax2.text(i, score + 0.005, f'{score:.3f}', ha='center', fontsize=9)
ax2.set_xticks(range(len(sorted_asr_methods)))
ax2.set_xticklabels(sorted_asr_methods, rotation=45, ha='right')
ax2.set_ylabel('ASR')
ax2.set_title('Comparison of Baseline and Ensemble Methods (ASR)')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Baseline Methods'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Ensemble Methods')
]
ax1.legend(handles=legend_elements, loc='upper right')
ax2.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('comparison_plot.jpg', format='jpg')

plt.show()