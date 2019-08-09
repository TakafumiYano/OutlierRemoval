import re
import sys
import os
import numpy as np
import scipy
from sklearn.covariance import MinCovDet
import scipy.stats as stats

# import pdb; pdb.set_trace()

THRESHOLD_ALPHA = 0.001
THRESHOLD_ALPHA_MD_2 = 13.8155
THRESHOLD_ALPHA_MD_3 = 16.2662
THRESHOLD_ALPHA_NORMAL = 3.0943
# THRESHOLD_ALPHA = 0.01
# THRESHOLD_ALPHA_MD_2 = 8.39941
# THRESHOLD_ALPHA_MD_3 = 10.4650
# THRESHOLD_ALPHA_NORMAL = 2.3301


def omnibus_normality_test(s):

    # test the normality of the variables x
    # import pdb;pdb.set_trace()
    b1 = scipy.stats.skew(s)
    b2 = scipy.stats.kurtosis(s)
    n = s.shape[0]
    # calculating the zb1
    y = b1 * np.sqrt((n+1) * (n+3) / 6 / (n-2))
    beta2 = 3 * (n ** 2 + 27 * n - 70) * (n+1) * (n+3) / (n-2) / (n+5) / (n+7) / (n+9)
    w2 = -1 + np.sqrt(2 * (beta2 - 1))
    delta = 1 / np.sqrt(np.log(np.sqrt(w2)))
    alpha = np.sqrt(2 / (w2 - 1))
    zb1 = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))

    # calculating the zb2
    eb2 = 3 * (n-1) / (n+1)
    varb2 = 24 * n * (n-2) * (n-3) / (n+1) / (n+1) / (n+3) / (n+5)
    x = (b2 - eb2) / np.sqrt(varb2)
    beta1 = 6 * (n**2 -5 * n + 2) / (n+7) / (n+9) * np.sqrt(6 * (n+3) * (n+5) / n / (n-2) / (n-3))
    A = 6 + 8 / beta1 * (2 / beta1 + np.sqrt(1 + 4 / (beta1 ** 2)))
    zb2 = ((1-2 / 9 / A ) - ((1 -2 / A) * (1 + x * np.sqrt(2 / (A -4)))) * np.sqrt(2 / 9 / A))

    k = zb1 ** 2 + zb2 ** 2
    score = k.real

    return score



def calcurate_mahalabinos_distance(X):

    robust_cov = MinCovDet().fit(X)
    score = robust_cov.mahalanobis(X)
    return score


def calc_z_score(x):

    median_val = np.median(x)
    mad = np.median(abs(x - median_val), axis=0)
    score = (x - median_val) / mad
    return score


def calc_sigma(x, dissociation):

    median_val = np.median(x)
    mad = np.median(abs(x - median_val), axis=0)
    threshold = median_val + mad * dissociation
    return threshold


def outlier_detection_routine_main(keep_flag, tdi_all, cur_all, length_all, is_normal_length, is_normal_tdi, is_normal_cur):
    keep_index = np.where(keep_flag == True)[0]
    n_samples = np.sum(keep_flag == True)
    length = length_all[keep_index]
    length = np.reshape(length, (n_samples, 1))
    tdi = tdi_all[keep_index]
    tdi = np.reshape(tdi, (n_samples, 1))
    cur = cur_all[keep_index]
    cur = np.reshape(cur, (n_samples, 1))

    # Calculate the Track Density and Detect Outliers
    # tdi_threshold = calc_sigma(tdi, -3)
    tdi_outliers = tdi < 3
    tdi_exclusion_index = keep_index[np.where(tdi_outliers > 0)[0]]
    # Calculate the Mahalanobis Distances and Detect Outliers
    Xtemp = np.hstack((tdi, cur, length))
    remove_column = []
    if is_normal_length == 0:
        remove_column.append(2)
    if is_normal_tdi == 0:
        remove_column.append(0)
    if is_normal_cur == 0:
        remove_column.append(1)
    # we define which row or column is deleted by axis boolean number (column is that axis = 1).
    X = np.delete(Xtemp, remove_column, 1)
    remain_column = 3 - len(remove_column)
    if remain_column == 3:
        robust_mahalanobis = calcurate_mahalabinos_distance(X)
        robust_mahalanobis = np.reshape(robust_mahalanobis, (n_samples, 1))
        outliers = robust_mahalanobis > THRESHOLD_ALPHA_MD_3
        outlier_index = keep_index[np.where(outliers > 0)[0]]
    elif remain_column == 2 :
        robust_mahalanobis = calcurate_mahalabinos_distance(X)
        robust_mahalanobis = np.reshape(robust_mahalanobis, (n_samples, 1))
        outliers = robust_mahalanobis > THRESHOLD_ALPHA_MD_2
        outlier_index = keep_index[np.where(outliers > 0)[0]]
    elif remain_column == 1:
        upper_threshold = calc_sigma(X, THRESHOLD_ALPHA_NORMAL)
        lower_threshold = calc_sigma(X, -THRESHOLD_ALPHA_NORMAL)
        outliers = np.logical_or(X < lower_threshold, X > upper_threshold)
        outlier_index = keep_index[np.where(outliers > 0)[0]]
    else:
        outlier_index = []
    # save the result flags
    new_keep_flag = keep_flag
    new_keep_flag[tdi_exclusion_index] = 0
    new_keep_flag[outlier_index] = 0

    return new_keep_flag


def outlier_detection_routine(keep_flag, tdi_all, cur_all, length_all):
    keep_index = np.where(keep_flag == True)[0]
    # length
    length = length_all[keep_index]
    non_normality_length = omnibus_normality_test(length)
    # density
    tdi = tdi_all[keep_index]
    non_normality_tdi = omnibus_normality_test(tdi)
    # curvature
    cur = cur_all[keep_index]
    non_normality_cur = omnibus_normality_test(cur)
    print("non-normality-length " + str(non_normality_length))
    print("non-normality-densitiy " + str(non_normality_tdi))
    print("non-normality-curvature " + str(non_normality_cur))
    is_normal_length = non_normality_length < THRESHOLD_ALPHA_MD_2
    is_normal_tdi = non_normality_tdi < THRESHOLD_ALPHA_MD_2
    is_normal_cur = non_normality_cur < THRESHOLD_ALPHA_MD_2
    new_keep_flag = outlier_detection_routine_main(keep_flag, tdi_all, cur_all, length_all, is_normal_length, is_normal_tdi, is_normal_cur)
    return new_keep_flag


def main(args):
    target_file_dir = args[1]
    tckdict, tckfile = os.path.split(target_file_dir)
    tckname, tckext = os.path.splitext(tckfile)

    length_stats = tckname+"-stat.txt"
    tdi_stats = tckname+"-tdi.txt"
    cur_stats = tckname+"-cur.txt"
    length = np.loadtxt(length_stats, delimiter = '\t')
    tdi = np.loadtxt(tdi_stats, delimiter = '\t')
    cur = np.loadtxt(cur_stats, delimiter = '\t')
    n_samples = length.shape[0]
    median_tdi = np.median(tdi)

    # 初期化変数
    keep_flag_arr = np.ones(n_samples)
    nkeep_fibers_prev = np.sum(keep_flag_arr)
    print(nkeep_fibers_prev)

    for i in range(10):
        if n_samples < 40 or median_tdi < 3:
            print("too tiny or too sparse fiber bundle!")
            break
        try:
            new_keep_flag_arr = outlier_detection_routine(keep_flag_arr, tdi, cur, length)
            keep_flag_arr = new_keep_flag_arr
            nkeep_fibers = np.sum(keep_flag_arr)
            print(nkeep_fibers)
        except:
            print("Exceptions occurred")
        if nkeep_fibers_prev - nkeep_fibers < 1:
            print("converged!")
            break
        else:
            nkeep_fibers_prev = nkeep_fibers

    keep_fiber_name = tckname + "-keep_fibers.txt"
    np.savetxt(keep_fiber_name, keep_flag_arr)


if __name__ == '__main__':
    main(sys.argv)

