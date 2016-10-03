from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from itertools import imap, combinations

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


# #---------------------------------------------------------------
# # Histogram plotting
# #---------------------------------------------------------------
# print("""\n
# #---------------------------------------------------------------
# # Plotting histograms...
# #---------------------------------------------------------------
# """)
# # Plot histogram for attributes
# for attribute in range(trainset_x.shape[1]):
#     plt.hist(trainset_x[:, attribute], bins=10)
#     plt.title("Histogram for attribute %d" %(attribute + 1))
#     plt.show()
#
# plt.hist(trainset_y, bins=10)
# plt.title("Histogram for attribute 14 (target)")
# plt.xlabel('Attribute values')
# plt.ylabel('Frequency')
# plt.show()

#---------------------------------------------------------------
# Pearson Correlations
#---------------------------------------------------------------
print("""\n
#---------------------------------------------------------------
# Pearson Correlations
#---------------------------------------------------------------
""")

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

pearson_correlations = []

# Find Pearson correlation of each attr with target value
for attribute in range(trainset_x.shape[1]):
    pearson_correlation = pearsonr(trainset_x[:, attribute], trainset_y)
    print("Pearson correlation of attribute %d with target value" % (attribute + 1))
    print(pearson_correlation)
    pearson_correlations.append(pearson_correlation)

print("Vector with pearson_correlations")
print(pearson_correlations)


#---------------------------------------------------------------
# Linear Regression
#---------------------------------------------------------------
print("""\n
#---------------------------------------------------------------
# Linear Regression
#---------------------------------------------------------------
""")

# normalize data - Z score normalization
def normalize_zscore(x, mean_vector, std_vector):
    return (x - mean_vector) / std_vector

def normalize_zscore_columns(trainset_x):
    # Calculate mean and std of each column
    mean_vector = trainset_x.mean(axis=0)     # to take the mean of each col
    std_vector = trainset_x.std(axis=0)     # to take the mean of each col
    # # For debugging purposes
    # print "Mean Vector"
    # print mean_vector
    # print "Std Vector"
    # print std_vector

    # normalize data
    normal_trainset_x = normalize_zscore(trainset_x, mean_vector, std_vector)
    return (normal_trainset_x, mean_vector, std_vector)

# normalize data
(normal_trainset_x, mean_vector, std_vector) = normalize_zscore_columns(trainset_x)


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
# np.savetxt('augmented_normal_trainset_x.csv', augmented_normal_trainset_x, delimiter=',')

#---------------------------------------------------------------
# We get w. This is how we train and obtain our linear regressor
#---------------------------------------------------------------
def find_w_param(augmented_normal_trainset_x, trainset_y):
    aug_w = np.dot(
        np.linalg.pinv(
            np.dot(augmented_normal_trainset_x.transpose(), augmented_normal_trainset_x)
        ),
        np.dot(augmented_normal_trainset_x.transpose(), trainset_y)
    )
    return aug_w

aug_w = find_w_param(augmented_normal_trainset_x, trainset_y)

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
print "(Linear Regression) MSE training set: %f" %mse_trainset


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
print "(Linear Regression) MSE testing set: %f" %mse_testset



#---------------------------------------------------------------
# Ridge Regression
#---------------------------------------------------------------
print("""\n
#---------------------------------------------------------------
# Ridge Regression
#---------------------------------------------------------------
""")

# Now let's find the parameter and complete the same process for Ridge regression

for param_lambda in [0.01, 0.1, 1.0]:
    #---------------------------------------------------------------
    # We get w. This is how we train and obtain our ridge regressor
    #---------------------------------------------------------------
    ridge_aug_w = np.dot(
        np.linalg.pinv(
            np.dot(augmented_normal_trainset_x.transpose(), augmented_normal_trainset_x) +
            param_lambda * np.identity(int(augmented_normal_trainset_x.shape[1]))
        ),
        np.dot(augmented_normal_trainset_x.transpose(), trainset_y)
    )

    # Use our trained algorithm to predict y and compare to the real y - find MSE
    mse_trainset = find_MSE(ridge_aug_w, augmented_normal_trainset_x, trainset_y)
    print "(Ridge Regression, lambda = %f) MSE training set: %f" % (param_lambda, mse_trainset)

    # FOR TEST SET: Use our trained algorithm to predict y and compare to the real y - find MSE
    mse_testset = find_MSE(ridge_aug_w, augmented_normal_testset_x, testset_y)
    print "(Ridge Regression, lambda = %f) MSE testing set: %f" % (param_lambda, mse_testset)


# #---------------------------------------------------------------
# # Ridge Regression with Cross-Validation
# #---------------------------------------------------------------
# print("""\n
# #---------------------------------------------------------------
# # Ridge Regression with Cross-Validation
# #---------------------------------------------------------------
# """)
# for hyperparam_lambda in [0.0001, 0.001, 0.01, 0.1, 1.0]:
#     print hyperparam_lambda




#---------------------------------------------------------------
# TOP 4 FEATURES CORRELATED WITH TARGET
#---------------------------------------------------------------
print("""\n
#---------------------------------------------------------------
# TOP 4 FEATURES CORRELATED WITH TARGET
#---------------------------------------------------------------
""")
abs_pearson_correlations = map(abs, pearson_correlations)

# For debugging purposes
# print "Absolute values of Pearson correlations"
# print abs_pearson_correlations
# print "Sorted absolute values of Pearson correlations"
# print sorted(abs_pearson_correlations, reverse=True)
print "TOP 4 attributes with highest correlations with target (attributes are counted from 0)"
top_4_features = np.argsort(abs_pearson_correlations)[::-1][:4]
# For debugging purposes
print top_4_features


columns_to_select = np.append(0, top_4_features + 1) # DON'T FORGET TO ADD THE FIRST COLUMN - WE NEED AN AUGMENTED X!!!!
highest4_augmented_normal_trainset_x = augmented_normal_trainset_x[:, columns_to_select]
# # For debugging purposes
# np.savetxt('highest4_augmented_normal_trainset_x.csv', highest4_augmented_normal_trainset_x, delimiter=',')


# TRAINING SET: Use our trained algorithm to predict y and compare to the real y - find MSE
highest4_aug_w = find_w_param(highest4_augmented_normal_trainset_x, trainset_y)
mse_trainset = find_MSE(highest4_aug_w, highest4_augmented_normal_trainset_x, trainset_y)
print "(Linear Regression, 4 features w/ highest correlation w/ target) MSE training set: %f" %mse_trainset

# FOR TEST SET
highest4_augmented_normal_testset_x = augmented_normal_testset_x[:, columns_to_select]
# FOR TEST SET: Use our trained algorithm to predict y and compare to the real y - find MSE
mse_testset = find_MSE(highest4_aug_w, highest4_augmented_normal_testset_x, testset_y)
print "(Linear Regression, 4 features w/ highest correlation w/ target) MSE testing set: %f" %mse_testset


#---------------------------------------------------------------
# ITERATIVE ADDING TOP FEATURE CORRELATED WITH RESIDUE
#---------------------------------------------------------------
print("""\n
#---------------------------------------------------------------
# ITERATIVE ADDING TOP FEATURE CORRELATED WITH RESIDUE
#---------------------------------------------------------------
""")

def find_residue(aug_w, aug_x, y):
    predicted_values_vector = linear_regression_prediction(aug_w, aug_x)
    diff_v = predicted_values_vector - y
    return diff_v

def pearson_r_list(matrix_x, vector_y):
    pearson_correlations = []
    # Find Pearson correlation of each attr with target value
    for attribute in range(matrix_x.shape[1]):
        pearson_correlation = pearsonr(matrix_x[:, attribute], vector_y)
        pearson_correlations.append(pearson_correlation)
    return pearson_correlations


top_features = []
top_feature = np.argsort(abs_pearson_correlations)[::-1][0] + 1 #DON'T FORGET TO ADD 1: AUGMENTED X, COLUMN 0 IS VECTOR 1
top_features.append(top_feature)
# print "top_feature %d" %top_feature
columns_to_select = np.array([0]) # DON'T FORGET TO ADD THE FIRST COLUMN - WE NEED AN AUGMENTED X!!!!

for i in range(4):
    print "Adding feature %d" %(top_feature)
    columns_to_select = np.append(columns_to_select, top_feature)
    selected_aug_norm_x = augmented_normal_trainset_x[:, columns_to_select]

    residue_trainset = find_residue(find_w_param(selected_aug_norm_x, trainset_y), selected_aug_norm_x, trainset_y)
    # print "Residue:", residue_trainset
    # print "Residue shape:", residue_trainset.shape

    residues_per_feature = map(abs, pearson_r_list(normal_trainset_x, residue_trainset))
    features_ordered_desc_residues = [attr for attr in np.argsort(residues_per_feature) if attr not in top_features][::-1]
    top_feature = features_ordered_desc_residues[0] + 1 #DON'T FORGET TO ADD 1: AUGMENTED X, COLUMN 0 IS VECTOR 1
    # print "top_feature %d" %top_feature

    # For debugging purposes
    print "columns_to_select", columns_to_select

print "Attributes seleected", columns_to_select[1:]

# For debugging purposes
# print "selected_aug_norm_x", selected_aug_norm_x

# TRAINING SET: Use our trained algorithm to predict y and compare to the real y - find MSE
iterative_top4_aug_w = find_w_param(selected_aug_norm_x, trainset_y)
mse_trainset = find_MSE(iterative_top4_aug_w, selected_aug_norm_x, trainset_y)
print "(Linear Regression, iterative top 4 features [highest correlation w/ residue]) MSE training set: %f" %mse_trainset

# FOR TEST SET
iterative_top4_augmented_normal_testset_x = augmented_normal_testset_x[:, columns_to_select]
# FOR TEST SET: Use our trained algorithm to predict y and compare to the real y - find MSE
mse_testset = find_MSE(iterative_top4_aug_w, iterative_top4_augmented_normal_testset_x, testset_y)
print "(Linear Regression, iterative top 4 features [highest correlation w/ residue]) MSE testing set: %f" %mse_testset


#---------------------------------------------------------------
# Selection with Brute-force search
#---------------------------------------------------------------
print("""\n
#---------------------------------------------------------------
# Selection with Brute-force search
#---------------------------------------------------------------
""")
feature_combinations = list([[0] + # don't forget to add the augmentation column!
                            # all possible combinations of attributes [1, 13]
                             list(x) for x in combinations(range(1, augmented_normal_trainset_x.shape[1]),4)])
# print feature_combinations
# print len(tuple(combinations(range(normal_trainset_x.shape[1]),4)))
best_features_combination = feature_combinations[0]
mse_trainset_best_combination = mse_testset_best_combination = np.inner(testset_y.transpose(), testset_y) #some really big number as initial value for MSE
# For debugging purposes
# print "initial mse_trainset_best_combination: ", mse_trainset_best_combination
# print "initial mse_testset_best_combination: ", mse_testset_best_combination

for columns_to_select in feature_combinations:
    selected_augmented_normal_trainset_x = augmented_normal_trainset_x[:, columns_to_select]
    # For debugging purposes
    # print selected_augmented_normal_trainset_x.shape
    # print str(columns_to_select)
    # np.savetxt('selected_augmented_normal_trainset_x.csv', selected_augmented_normal_trainset_x, delimiter=',')
    #
    #
    # TRAINING SET: Use our trained algorithm to predict y and compare to the real y - find MSE
    selected_aug_w = find_w_param(selected_augmented_normal_trainset_x, trainset_y)
    mse_trainset = find_MSE(selected_aug_w, selected_augmented_normal_trainset_x, trainset_y)
    # print "(Linear Regression, brute force) MSE training set: %f" % mse_trainset

    # FOR TEST SET
    selected_augmented_normal_testset_x = augmented_normal_testset_x[:, columns_to_select]
    # FOR TEST SET: Use our trained algorithm to predict y and compare to the real y - find MSE
    mse_testset = find_MSE(selected_aug_w, selected_augmented_normal_testset_x, testset_y)
    # print "(Linear Regression, brute force) MSE testing set: %f" % mse_testset

    if mse_testset < mse_testset_best_combination:
        mse_trainset_best_combination = mse_trainset
        mse_testset_best_combination = mse_testset
        best_features_combination = columns_to_select[1:]

print "Best combination of features: ", best_features_combination
print "(Linear Regression, Brute-force search, Best combination of features) MSE training set: %f" %mse_trainset_best_combination
print "(Linear Regression, Brute-force search, Best combination of features) MSE testing set: %f" %mse_testset_best_combination




#---------------------------------------------------------------
# Polynomial Feature Expansion
#---------------------------------------------------------------
print("""\n
#---------------------------------------------------------------
# Polynomial Feature Expansion
#---------------------------------------------------------------
""")

def expand_features(trainset_x):
    # squared_trainset_x = np.square(trainset_x)
    # print "squared_trainset_x.shape:", squared_trainset_x.shape
    #     # np.dot(trainset_x.transpose(), )
    # count = 0
    expanded_trainset_x = np.empty([trainset_x.shape[0], 0])
    # print expanded_trainset_x.shape
    # np.hstack([np.array((int(), 1)), unaugmented_x])
    #
    for i in range(trainset_x.shape[1]):
        for j in range(i, trainset_x.shape[1]):
            # print (i, j)
            # count += 1
            new_column = np.multiply(trainset_x[:, i], trainset_x[:, j])
            # print new_column
            # print new_column.shape
            expanded_trainset_x = np.column_stack((expanded_trainset_x, new_column))

    # For debuggin purposes
    # print count
    # print "expanded_trainset_x", expanded_trainset_x
    # print "expanded_trainset_x.shape", expanded_trainset_x.shape
    return expanded_trainset_x

# get new features
expanded_trainset_x = expand_features(trainset_x)

# normalize new features
(expanded_normal_trainset_x, mean_vector_expanded_trainset_x, std_vector_expanded_trainset_x) = normalize_zscore_columns(expanded_trainset_x)

# For debugging purposes
# print "expanded_normal_trainset_x.mean", expanded_normal_trainset_x.mean(axis=0)
# print "expanded_normal_trainset_x.std", expanded_normal_trainset_x.std(axis=0)
# print "expanded_normal_trainset_x.mean", expanded_normal_trainset_x.mean(axis=0).shape
# print "expanded_normal_trainset_x.std", expanded_normal_trainset_x.std(axis=0).shape

# now add old features and new features together
expanded_normal_trainset_x = np.column_stack((normal_trainset_x, expanded_normal_trainset_x))
# print "expanded_trainset_x.shape:", expanded_trainset_x.shape

# # For debugging purposes
# print "expanded_normal_trainset_x.mean", expanded_normal_trainset_x.mean(axis=0)
# print "expanded_normal_trainset_x.std", expanded_normal_trainset_x.std(axis=0)
# print "expanded_normal_trainset_x.mean", expanded_normal_trainset_x.mean(axis=0).shape
# print "expanded_normal_trainset_x.std", expanded_normal_trainset_x.std(axis=0).shape

# Augment X with vector 1
expanded_augmented_normal_trainset_x = augment_x(expanded_normal_trainset_x)
# print "expanded_augmented_normal_trainset_x"
# print expanded_augmented_normal_trainset_x[:, 0]

expanded_aug_w = find_w_param(expanded_augmented_normal_trainset_x, trainset_y)
# Use our trained algorithm to predict y and compare to the real y - find MSE
mse_trainset = find_MSE(expanded_aug_w, expanded_augmented_normal_trainset_x, trainset_y)
print "(Linear Regression, Polynomial Feature Expansion) MSE training set: %f" %mse_trainset

# FOR TEST SET
# get new features
expanded_testset_x = expand_features(testset_x)
# normalize new features
expanded_normal_testset_x = normalize_zscore(expanded_testset_x, mean_vector_expanded_trainset_x, std_vector_expanded_trainset_x)
# now add old features and new features together
expanded_normal_testset_x = np.column_stack((normal_testset_x, expanded_normal_testset_x))
# Augment X with vector 1
expanded_augmented_normal_testset_x = augment_x(expanded_normal_testset_x)
# FOR TEST SET: Use our trained algorithm to predict y and compare to the real y - find MSE
mse_testset = find_MSE(expanded_aug_w, expanded_augmented_normal_testset_x, testset_y)
print "(Linear Regression, Polynomial Feature Expansion) MSE testing set: %f" %mse_testset





