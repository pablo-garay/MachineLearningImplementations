import scipy.io
from svmutil import *
import numpy as np
import timeit

train = scipy.io.loadmat('phishing-train.mat')
test = scipy.io.loadmat('phishing-test.mat')

train_features = train['features']
train_label = train['label']
test_features = test['features']
test_label = test['label']

# debugging
# print train_features.shape
# print train_label.shape
# print test_features.shape
# print test_label.shape
# print type(train_features)


# print sum(new_train_features == train_features[:,[0]])
# print new_train_features.shape

np.set_printoptions(threshold=np.inf)

# Do Preprocessing
def features_data_preprocessing(data_features):
    new_data_features = np.empty([data_features.shape[0], 0])
    # print new_data_features.shape

    for num_col in range(data_features.shape[1]):
        if all(x in data_features[:, [num_col]] for x in [-1, 0, 1]): # feature fi have value [-1, 0, 1], we create 3 new features
            col = data_features[:, num_col]
            new_columns = np.empty([data_features.shape[0], 3])

            for (index, val) in enumerate(col):
                if val ==  -1: new_columns[index] = [1, 0, 0]
                elif val == 0: new_columns[index] = [0, 1, 0]
                elif val == 1: new_columns[index] = [0, 0, 1]
            # print new_columns
            # print "\n\n"
            new_data_features = np.hstack((new_data_features, new_columns))

        elif all(x in data_features[:, [num_col]] for x in [-1, 1]): # transform any feature of [-1, 1] to [0, 1]
            # replace -1s for 0s
            col = data_features[:, [num_col]]
            col[col == -1] = 0
            # print col
            new_data_features = np.hstack((new_data_features, col))
            # print train_features[:, [num_col]]
        else:
            # no problem, just add column as it is
            new_data_features = np.hstack((new_data_features, data_features[:, [num_col]]))

    return new_data_features

new_train_features = features_data_preprocessing(train_features)
new_test_features = features_data_preprocessing(test_features)
# print new_train_features.shape
# print new_test_features.shape
# print train_label.shape
# print train_features.shape

def train_time(stmt='pass', setup='pass'):
    print "Average training time: %f seconds" %timeit.Timer(stmt, setup).timeit(1)
    return


# Use SVM
# print "train_label: ", train_label, "train_features: ", train_features, train_label[0]
prob = svm_problem(train_label[0], # formatting for correct paramater passing
                   train_features[:, :].tolist())

def svm_set_params(kernel_type, C, cross_validation = 1, nr_fold = 3):
    # set parameters for SVM
    param = svm_parameter()

    # kernel_type : set type of kernel function (default 2)
    # 	0 -- linear: u'*v
    # 	1 -- polynomial: (gamma*u'*v + coef0)^degree
    # 	2 -- radial basis function: exp(-gamma*|u-v|^2)
    # 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
    param.kernel_type = kernel_type
    param.cross_validation = cross_validation  # use cross_validation
    param.C = C  # cost : set the parameter C
    param.nr_fold = nr_fold  # 3-fold cross validation
    param.print_func = PRINT_STRING_FUN(print_null)
    return param

for type_kernel in [LINEAR, POLY, RBF]:
    if type_kernel == LINEAR:
        list_exp_c = range(-6, 2 + 1)
        for exp_c in list_exp_c:
            C = 4 ** exp_c
            # print C
            param = svm_set_params(type_kernel, C, cross_validation = 1, nr_fold = 3)
            # m=svm_train(prob, param)
            # print "\n\n"
            print "\nLinear SVM (C = 4 ** %d)" % exp_c
            train_time('m=svm_train(prob, param)', "from __main__ import svm_train, prob, param")

    else:
        list_exp_c = range(-3, 7 + 1)
        for exp_c in list_exp_c:
            C = 4 ** exp_c
            # print C
            param = svm_set_params(type_kernel, C, cross_validation = 1, nr_fold = 3)

            if type_kernel == POLY: #polynomial, set degree param
                for degree in [1, 2, 3]:
                    param.degree = degree
                    print "\nSVM, Polynomial Kernel (C = 4 ** %d, degree = %d)" % (exp_c, degree)
                    train_time('m=svm_train(prob, param)', "from __main__ import svm_train, prob, param")

            elif type_kernel == RBF: #RBF or gaussian, set gamma param
                list_exp_gamma = range(-7, -1 + 1)
                for exp_gamma in list_exp_gamma:
                    gamma = 4 ** exp_gamma
                    param.gamma = gamma
                    print "\nSVM, RBF Kernel (C = 4 ** %d, gamma = 4 ** %d)" % (exp_c, exp_gamma)
                    train_time('m=svm_train(prob, param)', "from __main__ import svm_train, prob, param")

            # m.predict([1,1,1])
            # pm = svm_parameter(kernel_type=RBF)
