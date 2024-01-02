import matplotlib.pyplot as plt
import numpy as np

# Define the perceptron function
def perceptron(x1, x2, w1, w2, b):
    # Calculate the weighted sum
    weighted_sum = w1 * x1 + w2 * x2 + b

    # Apply the activation function
    if weighted_sum >= 0:
        return 1
    else:
        return 0

# Set the weights and bias
w1 = 1
w2 = -1
b = -0.5

# Generate data points
x1 = np.array([-1, -0.5, 0, 0.5, 1])
x2 = np.array([-1, 0, 0.5, 1, 1.5])
y = np.array([0, 0, 0, 1, 1])

# Calculate the predicted values
predicted_values = []
for i in range(len(x1)):
    predicted_value = perceptron(x1[i], x2[i], w1, w2, b)
    predicted_values.append(predicted_value)

# Plot the data points and decision boundary
plt.figure()
plt.scatter(x1, x2, c=y)

# Calculate the slope and intercept of the decision boundary
m = -w2 / w1
c = -b / w1

# Generate points on the decision boundary
x_min = -1.5
x_max = 2.5
y_min = m * x_min + c
y_max = m * x_max + c

decision_boundary = np.array([
    [x_min, y_min],
    [x_max, y_max]
])

plt.plot(decision_boundary[:, 0], decision_boundary[:, 1], color='green')

# Draw weights and bias
plt.plot([-0.5, 1.5], [0, 0], color='black', linestyle='--', label='w1')
plt.plot([0, 0], [-0.5, 1.5], color='black', linestyle='--', label='w2')
plt.scatter([-0.5, 1.5], [0, 0], color='black', marker='o', label='b')

# Show the plot
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.show()
