import numpy as np
import seaborn as sns
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def cbs_stat(x, start, end):
    x0 = x[start:end] - np.mean(x[start:end])
    n = end - start
    y = np.cumsum(x0)
    e0, e1 = np.argmin(y), np.argmax(y)
    i0, i1 = min(e0,e1), max(e0, e1)
    s0, s1 = y[i0], y[i1]
    return (s1-s0)**2*n/(i1-i0)/(n-i1+i0), i0+start, i1+start+1


def t_single(x, start, end, i):
    s0 = np.mean(x[start:i])
    s1 = np.mean(x[i:end])
    t = (s0-s1)**2*(i-start)*(end-i)/(end-start)
    return t


def cbs(x, shuffles=1000):
    start=0
    end=len(x)
    max_t, max_start, max_end = cbs_stat(x ,start,end)
    if max_end-max_start == len(x):
        return False, max_t, max_start, max_end
    thresh_count=0
    alpha = shuffles*.05
    for i in range(shuffles):
        threshold = cbs_stat(np.random.permutation(x[start:end]),0,end-start)[0]
        if threshold >= max_t:
            thresh_count+=1
        if thresh_count > alpha:
            return False, max_t, max_start, max_end
    return True, max_t, max_start, max_end


def segment(x, start, end, L=[]):
    threshold, t, s, e = cbs(x[start:end])
    log.info('Proposed partition of {} to {} from {} to {} with t value {} is {}'.format(start, end, start+s, start+e,t,threshold))
    if not threshold  :
        L.append((start,end))
    else:
        if s>5:
            segment(x, start, start+s, L)
        if e-s>5:
            segment(x, start+s, start+e, L)
        if  e<end-start-5:
            segment(x, start+e, end, L)
    return L

    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    N = 100
    means = np.random.randint(-5, 5, N)
    runs = np.random.randint(10, 50, N)
    sample = np.concatenate([([x]*y)+np.random.normal(0,.5,size=y) for (x,y) in zip(means,runs)])
    sample = sample 

    L = segment(sample, 0, len(sample))
    j=sns.scatterplot(list(range(len(sample))),sample)
    for x in L:
        j.axvline(x[0])
    j.axvline(L[-1][1])
    j.axvline(0)
    j.axvline(np.cumsum(runs)[-1])
    x0=0
    
    for i,x in enumerate(np.cumsum(runs)):
        j.hlines(means[i],x0-1, x-1 ,color='orange',linewidth=2.0)
        x0 = x
    f = j.get_figure()
    f.set_size_inches(8,8)
    f.savefig('plot.png')
