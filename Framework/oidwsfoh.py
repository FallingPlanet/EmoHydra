import matplotlib.pyplot as plt

# Given data for the bar chart
categories = ['Text Model Accuracy', 'Face Recognition Accuracy', 'Speech Recognition Accuracy']
values = [87, 72, 71]  # Corresponding accuracies

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['blue', 'green', 'red'])

# Add the percentage values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom')

# Set the title and labels
plt.title('Accuracy Comparison of Heterogenous Models')
plt.xlabel('Model Type')
plt.ylabel('Accuracy (%)')

# Set the y-axis limit to accommodate the text above the bars
plt.ylim(0, 100)

# Show the plot
plt.tight_layout()
plt.show()
