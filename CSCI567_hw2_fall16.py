from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from itertools import imap

boston = load_boston()

# For debugging purposes
# print(boston.data)
# print(boston.target)
# print(boston.data.shape)
# print(boston.target.shape)


testset_x = np.empty( shape=(0, 13) )
testset_y = np.empty( shape=(0) )
trainset_x = np.empty( shape=(0, 13) )
trainset_y = np.empty( shape=(0) )

# indexes = range(0, len(boston.data), 7)
# boston.data[::7]
# boston.target[::7]

# split data according. Test set consists of 7i-th points, the rest will be used as training set
for i in range(len(boston.data)):
    if i % 7 == 0:
        testset_x = np.vstack([testset_x, boston.data[i]])
        testset_y = np.append(testset_y, boston.target[i])
    else:
        trainset_x = np.vstack([trainset_x, boston.data[i]])
        trainset_y = np.append(trainset_y, boston.target[i])


# For debugging purposes
# print("testset_y")
# print(testset_y)
# print("testset_y shape")
# print(testset_y.shape)
#
# print("trainset_y")
# print(trainset_y)
# print("trainset_y shape")
# print(trainset_y.shape)
#
# print("testset_x.shape")
# print(testset_x.shape)
# print("trainset_x.shape")
# print(trainset_x.shape)

# For debugging purposes
# Testing and test cases...
# testset_x == boston.data[::7]
# testset_y == boston.target[::7]

# For debugging purposes
# print "Testing equality for Training set X......... (hit enter to continue)"
# raw_input()
# j = 0
# for i in range(len(boston.data)):
#     if i % 7 != 0:
#         print(boston.data[i] == trainset_x[j])
#         j += 1

# For debugging purposes
# print "Testing equality for Training set Y......... (hit enter to continue)"
# raw_input()
# j = 0
# for i in range(len(boston.data)):
#     if i % 7 != 0:
#         print(boston.target[i] == trainset_y[j])
#         j += 1

# For debugging purposes
# Save output to file
# np.savetxt('output_file.csv', trainset_x, delimiter=',')   # X is an array


# # Plot histogram for attributes
# for attribute in range(trainset_x.shape[1]):
#     plt.hist(trainset_x[:, attribute], bins=10)
#     plt.title("Histogram for attribute %d" %(attribute + 1))
#     plt.show()
#
# plt.hist(trainset_y, bins=10)
# plt.title("Histogram for attribute 14 (target)")
# plt.show()

# # Pearson correlation with numpy - Not allowed :(
# for attribute in range(trainset_x.shape[1]):
#     print("Pearson correlation of attribute %d with target value" % (attribute + 1))
#     print(np.corrcoef(trainset_x[:, attribute], trainset_y))[0, 1]

# Pearson correlation implementation
def pearsonr(x, y):
  # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(map(lambda x: pow(x, 2), x))
  sum_y_sq = sum(map(lambda x: pow(x, 2), y))
  psum = sum(imap(lambda x, y: x * y, x, y))
  num = psum - (sum_x * sum_y/n)
  den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if den == 0: return 0
  return num / den

# Find Pearson correlation of each attr with target value
for attribute in range(trainset_x.shape[1]):
    print("Pearson correlation of attribute %d with target value" % (attribute + 1))
    print(pearsonr(trainset_x[:, attribute], trainset_y))


# Calculate mean and std of each column
mean_vector = trainset_x.mean(axis=0)     # to take the mean of each col
std_vector = trainset_x.std(axis=0)     # to take the mean of each col
# # For debugging purposes
# print "Mean Vector"
# print mean_vector
# print "Std Vector"
# print std_vector

# normalize data - Z score normalization
def normalize_zscore(x, mean_vector, std_vector):
    return (x - mean_vector) / std_vector

# normalize data
normal_trainset_x = normalize_zscore(trainset_x, mean_vector, std_vector)


# np.savetxt('trainset_x.csv', trainset_x, delimiter=',')
# np.savetxt('mean_vector.csv', mean_vector, delimiter=',')
# np.savetxt('std_vector.csv', std_vector, delimiter=',')
# np.savetxt('normal_trainset_x.csv', (trainset_x - mean_vector)/std_vector, delimiter=',')

# For debugging purposes
# A few test cases
# np.savetxt('test.csv', (np.zeros((433, 13)) - mean_vector) / mean_vector, delimiter=',')
# np.savetxt('test.csv', (np.ones((433, 13))) * np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) / std_vector, delimiter=',')
# np.savetxt('test.csv', (np.ones((433, 13))) * np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] - mean_vector) / std_vector, delimiter=',')
# normal_trainset_x.mean(axis=0)
# normal_trainset_x.std(axis=0)

# Augment X with vector 1
def augment_x(unaugmented_x):
    return np.hstack([np.ones((int(unaugmented_x.shape[0]), 1)), unaugmented_x])

# Augment X with vector 1
augmented_normal_trainset_x = augment_x(normal_trainset_x)
# # For debugging purposes
# np.savetxt('augmented_x.csv', augmented_normal_trainset_x, delimiter=',')

#---------------------------------------------------------------
# We get w. This is how we train and obtain our linear regressor
#---------------------------------------------------------------
aug_w = np.dot(
    np.linalg.pinv(
        np.dot(augmented_normal_trainset_x.transpose(), augmented_normal_trainset_x)
    ),
    np.dot(augmented_normal_trainset_x.transpose(), trainset_y)
)

def linear_regression_prediction(aug_w, vector_x):
    predicted_y = np.inner(aug_w.transpose(), vector_x)
    return predicted_y

# Use our trained algorithm to predict y and compare to the real y - find MSE
def find_MSE(aug_w, aug_x, y):
    predicted_values_vector = linear_regression_prediction(aug_w, aug_x)
    diff_v = predicted_values_vector - y
    return np.inner(diff_v.transpose(), diff_v) / diff_v.size

# Use our trained algorithm to predict y and compare to the real y - find MSE
mse_trainset = find_MSE(aug_w, augmented_normal_trainset_x, trainset_y)
print "MSE training set: %f" %mse_trainset


# For debugging purposes
# print "not augmented:"
# print testset_x
# print "augmented:"
# print augment_x(testset_x)

# normalize testing data
normal_testset_x = normalize_zscore(testset_x, mean_vector, std_vector)

# For debugging purposes
# print "normal_testset_x mean vector:"
# print normal_testset_x.mean(axis=0)
# print "normal_testset_x std vector:"
# print normal_testset_x.std(axis=0)

# Augment it
augmented_normal_testset_x = augment_x(normal_testset_x)

# For debugging purposes
# np.savetxt('augmented_normal_testset_x.csv', augmented_normal_testset_x, delimiter=',')

# FOR TEST SET: Use our trained algorithm to predict y and compare to the real y - find MSE
mse_testset = find_MSE(aug_w, augmented_normal_testset_x, testset_y)
print "MSE testing set: %f" %mse_testset





# # Now let's find the parameter and complete the same process for Ridge regression
#
# for param_lambda in [0.01, 0.1, 1.0]:
#     #---------------------------------------------------------------
#     # We get w. This is how we train and obtain our ridge regressor
#     #---------------------------------------------------------------
#     ridge_aug_w = np.dot(
#         np.linalg.pinv(
#             np.dot(augmented_normal_trainset_x.transpose(), augmented_normal_trainset_x) + param_lambda * np.identity()
#         ),
#         np.dot(augmented_normal_trainset_x.transpose(), trainset_y)
#     )

