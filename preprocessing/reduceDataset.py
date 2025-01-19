import pandas as pd

print("Loading dataset...")
df = pd.read_csv('datasets/original_super_sms_dataset.csv', encoding='ISO-8859-1')
df.dropna(inplace=True)

label_counts = df['Labels'].value_counts()
total_samples = 20000
num_labels = len(label_counts)
samples_per_label = total_samples // num_labels

# Sample the dataset to create a balanced dataset with 20,000 entries
print("Sampling dataset to create a balanced dataset with 20,000 entries...")
balanced_df = pd.concat([
    df[df['Labels'] == label].sample(n=samples_per_label, random_state=42)
    for label in label_counts.index
]).reset_index(drop=True)

final_label_counts = balanced_df['Labels'].value_counts()
print("Final label counts:", final_label_counts)

output_file = 'datasets/super_sms_dataset.csv'
balanced_df.to_csv(output_file, index=False)
print(f"Balanced dataset saved to {output_file}")