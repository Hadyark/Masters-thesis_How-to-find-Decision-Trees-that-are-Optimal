# Import the extension module hello.
import error_cython
import utils

import load_data
X, y, sensitive = load_data.lawsuit()
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = utils.train_test_split(1, X, y, sensitive)

d =error_cython.discr_add2([0], y_train, sensitive_train)
print(d)
