from timeit import default_timer as timer
from multiprocessing.pool import Pool

from util_ransac import perform_ransac
from util_matching import perform_match
from util_sift import precompute_sift, load_sift

if __name__ == '__main__':
    precompute_sift('S_BB_V1', 'PW_BB_V1')
    s_im, s_label, s_kp, s_des = load_sift('S_BB_V1_SIFT.npz')
    pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V1_SIFT.npz')

    time_start = timer()

    pool = Pool()
    s_idx = range(s_kp.shape[0])

    print('Begin pool work')
    pool.map(perform_match, s_idx)
    # pool.map(perform_ransac, s_idx)
    pool.close()
    pool.join()

    duration = timer() - time_start
    print("Program took %.3fs" % duration)