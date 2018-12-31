from functools import lru_cache
import numpy as np

def cbs_stat(x, start, end):

    @lru_cache()
    def c(i,j):
        if (i == -1) | (j == -1) | (j<=i):
            return 0
        else:
            return x[j-1]+c(i,j-1)

    N = end - start
    max_start = start
    max_end = end
    max_t = 0
    for seg_start in range(start,end):
        for seg_end in range(seg_start+1,end):
            l1 = seg_end - seg_start
            l2 = seg_start + N - seg_end
            s1 = c(seg_start, seg_end)
            s2 = c(seg_end, N) + c(0, seg_start)
            t = (s1/l1 - s2/l2)/np.sqrt(1/l1 + 1/l2)
            if t > max_t:
                max_t = t
                max_start = seg_start
                max_end = seg_end
    return max_t, max_start, max_end


def cbs(x, shuffles=100):
    start=0
    end=len(x)
    threshold = np.quantile([cbs_stat(np.random.permutation(x),start,end)[0] for i in range(shuffles)],.95)
    max_t, max_start, max_end = cbs_stat(y,start,end)
    if max_t > threshold:
        return max_t, max_start, max_end
    else:
        return None


