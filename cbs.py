import numpy as np
import seaborn as sns


def cbs_stat(x, start, end):
    N = end - start
    max_start = start
    max_end = end
    max_t = 0
    for seg_start in range(start,end):
        for seg_end in range(seg_start+1,end):
            l1 = seg_end - seg_start
            l2 = seg_start + N - seg_end
            s1 = np.sum(x[seg_start:seg_end])
            s2 = np.sum(x[seg_end:N]) + np.sum(x[:seg_start])
            t = (s1/l1 - s2/l2)/np.sqrt(1/l1 + 1/l2)
            if t > max_t:
                max_t = t
                max_start = seg_start
                max_end = seg_end
    return max_t, max_start, max_end


def cbs(x, shuffles=100):
    start=0
    end=len(x)
    threshold = np.percentile([cbs_stat(np.random.permutation(x[start:end]),0,end-start)[0] for i in range(shuffles)],95)
    max_t, max_start, max_end = cbs_stat(x ,start,end)
    return threshold, max_t, max_start, max_end


def segment(x, start, end, L):
    threshold, t, s, e = cbs(x[start:end])
    print(start, end, threshold, t, s, e)
    if t < threshold:
        L.append((start,end))
    else:
        if s>0:
            segment(x,start, start+s,L)
        if e-s>1:
            segment(x,start+s, start+e,L)
        if start+e < end:
            segment(x,start+e,end,L)
        return L
    

if __name__ == '__main__':
    means = np.random.randint(-5, 5, 10)
    runs = np.random.randint(3, 30, 10)
    sample = np.concatenate([([x]*y)+np.random.normal(0,.2,size=y) for (x,y) in zip(means,runs)])
    base_line = np.concatenate([[x]*y for (x,y) in zip(means,runs)])
    L=[]
    segment(sample, 0, len(sample), L)
    j=sns.scatterplot(list(range(len(sample))),sample)
    for x in L:
        j.axvline(x[0])
    j.axvline(L[-1][1])
    j.axvline(0)
    x0=0

    for i,x in enumerate(np.cumsum(runs)):
        j.hlines(means[i],x0,x,color='orange')
        x0 = x

        
    f = j.get_figure()
    f.set_size_inches(8,8)
    f.savefig('plot.png')
