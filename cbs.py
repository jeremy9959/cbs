import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.WARN)


def cbs_stat(x):
    '''Given x, Compute the subinterval x[i0:i1] with the maximal segmentation statistic t. 
    Returns t, i0, i1'''
    
    x0 = x - np.mean(x)
    n = len(x0)
    y = np.cumsum(x0)
    e0, e1 = np.argmin(y), np.argmax(y)
    i0, i1 = min(e0,e1), max(e0, e1)
    s0, s1 = y[i0], y[i1]
    return (s1-s0)**2*n/(i1-i0)/(n-i1+i0), i0, i1


def cbs(x, shuffles=1000, p=.05):
    '''Given x, find the interval x[i0:i1] with maximal segmentation statistic t. Test that statistic against
    given (shuffles) number of random permutations with significance p.  Return True/False, t, i0, i1; True if
    interval is significant, false otherwise.'''
    
    max_t, max_start, max_end = cbs_stat(x)
    if max_end-max_start == len(x):
        return False, max_t, max_start, max_end
    thresh_count=0
    alpha = shuffles*p
    xt = x.copy()
    for i in range(shuffles):
        np.random.shuffle(xt)
        threshold, s0, e0  = cbs_stat(xt)
        if threshold >= max_t:
            thresh_count += 1
        if thresh_count > alpha:
            return False, max_t, max_start, max_end
    return True, max_t, max_start, max_end


def rsegment(x, start, end , L=[], shuffles=1000, p=.05):
    '''Recursively segment the interval x[start:end] returning a list L of pairs (i,j) where each (i,j) is a significant segment.
    '''
    threshold, t, s, e = cbs(x[start:end], shuffles=shuffles, p=p)
    log.info('Proposed partition of {} to {} from {} to {} with t value {} is {}'.format(start, end, start+s, start+e,t,threshold))
    if not threshold  :
        L.append((start,end))
    else:
        if s>5:
            rsegment(x, start, start+s, L)
        if e-s>5:
            rsegment(x, start+s, start+e, L)
        if  e<end-start-5:
            rsegment(x, start+e, end, L)
    return L


def segment(x, shuffles=1000, p=.05):
    '''Segment the array x, using significance test based on shuffles rearrangements and significance level p
    '''
    start = 0
    end = len(x)
    L=[]
    rsegment(x, start, end, L, shuffles=shuffles, p=p)
    return L


def generate_normal_time_series(num, minl=50, maxl=1000):
    '''Generate a time series with num segments of minimal length minl and maximal length maxl.  Within a segment,
    data is normal with randomly chosen, normally distributed mean between -10 and 10, variance between 0 and 1.
    '''
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn()*10
        var = np.random.randn()*1
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return data


if __name__ == '__main__':

    sample = generate_normal_time_series(10)
    L = segment(sample)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(12,4)
    ax = sns.lineplot(list(range(len(sample))),sample, size=0.1, color='black',  legend=None, ax = ax)
    
    for x in L:
        ax.axvline(x[0],color='gray',alpha=0.5)
    ax.axvline(L[-1][1],color='gray', alpha=0.5)
    ax.set_title('Segmentation of random normally distributed time series\n Algorithm is (simplified) circular binary segmentation')

