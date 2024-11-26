import pandas as pd

# File paths
test_path = '/data/ephemeral/home/sujin/test_answer_gpt4o.csv'

predictions_path = '/data/ephemeral/home/sujin/outputs/output_bllosom_clena_prompt.csv'

# Load data
test = pd.read_csv(test_path)
predictions = pd.read_csv(predictions_path)

# Calculate overall accuracy
accuracy = (test['answer'] == predictions['answer']).mean()
print(f"Overall Accuracy: {accuracy:.4f}")

# Accuracy for 434 rows sampled with random seed 42
sampled_indices = test.sample(n=434, random_state=42).index
sampled_accuracy = (test.loc[sampled_indices, 'answer'] == predictions.loc[sampled_indices, 'answer']).mean()
print(f"Accuracy for 434 random rows (seed=42): {sampled_accuracy:.4f}")