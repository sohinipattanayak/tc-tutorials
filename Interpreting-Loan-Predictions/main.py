import numpy as np
from trustyai.model import Model
from trustyai.explainers import LimeExplainer
import matplotlib.pyplot as plt

# Define weights for the linear model. In a real-world scenario, these would come from training data.
weights = np.random.uniform(low=-5, high=5, size=5)
print(f"Weights for Features: {weights}")

# Simple linear model representing the loan decision process
def linear_model(x):
    return np.dot(x, weights)

model = Model(linear_model)

# Sample data for an applicant (random for this example)
# [Annual Income, Open Accounts, Late Payments, Debt-to-Income Ratio, Credit Inquiries]
applicant_data = np.random.rand(1, 5)
predicted_credit_score = model(applicant_data)

lime_explainer = LimeExplainer(samples=1000, normalise_weights=False)
lime_explanation = lime_explainer.explain(
    inputs=applicant_data,
    outputs=predicted_credit_score,
    model=model)

print(lime_explanation.as_dataframe())

print("Summary of the explanation:")
if predicted_credit_score > 0:
    print("The applicant is likely to be approved for a loan.")
else:
    print("The applicant is unlikely to be approved for a loan.")

print("Feature weights:")
for feature, weight in zip(["Annual Income", "Number of Open Accounts", "Number of times Late Payment in the past", "Debt-to-Income Ratio", "Number of Credit Inquiries in the last 6 months"], weights):
    print(f"{feature}: {weight:.2f}")

# Access the dataframe for the output
output_df = lime_explanation.as_dataframe()["output-0"]

# Now sort the dataframe
exp_df = output_df.sort_values(by="Saliency")

# Mapping for input indices to feature names
feature_mapping = {
    "input-0": "Annual Income",
    "input-1": "Number of Open Accounts",
    "input-2": "Number of times Late Payment in the past",
    "input-3": "Debt-to-Income Ratio",
    "input-4": "Number of Credit Inquiries in the last 6 months"
}

y_axis = list(exp_df['Saliency'])
x_axis = [feature_mapping[feature] for feature in exp_df['Feature']]
colors = []

# Find the feature with highest influence
most_influential_feature_value = max(y_axis, key=abs)

for bar in y_axis:
    if bar == most_influential_feature_value:
        colors.append("green")  # Highlighting the most influential feature with a different color
    elif bar < 0:
        colors.append("red")
    else:
        colors.append("blue")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor("#f2f2f2")
ax.bar(x_axis, height=y_axis, color=colors)
plt.title('LIME: Feature Impact for Loan Application')
plt.xticks(rotation=20, ha='right', fontsize=10)
plt.axhline(0, color="black")  # x-axis line
plt.tight_layout()  # Adjust layout for better visualization
plt.show()
