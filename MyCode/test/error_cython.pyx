import numpy as np
from cython cimport long
from cython cimport double


cpdef discr_add2 (list tids, double[:] y, long[:] sensitive):
    cdef int p0 = 0;
    cdef int p1 = 0;

    for i in tids:
        if sensitive[i] == 0.0:
            #if y[i] == 1.0:
            p0 += 1
        elif sensitive[i] == 1.0:
            #if y[i] == 1.0:
            p1 += 1
    cnt_unique = np.unique(sensitive, return_counts=True)[1]
    n_zero = cnt_unique[0]
    n_one = cnt_unique[1]

    if n_one == 0 and n_zero == 0:
        d = 0
    elif n_zero == 0:
        d = -(p1 / n_one)
    elif n_one == 0:
        d = p0 / n_zero
    else:
        d = (p0 / n_zero) - (p1 / n_one)

    return d