# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Excel file
df = pd.read_excel('result-analysis.xlsx', sheet_name='box', header=0)

# Set plot style
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5)

# Create a box plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, palette="Set3")

# Add labels and a title
ax.set_title('Signal Controller Performance Comparison', fontsize=18, y=1.05, fontweight='bold')
ax.set_ylabel('Average Person Delay (s)', fontsize=16)

# Customize x-tick labels
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right')

# Customize y-tick labels
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.tick_params(axis='y', which='major', labelsize=14)

# Add horizontal grid lines
ax.yaxis.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig('Box plot of average person delay.png', dpi=300)

# Show the plot
plt.show()