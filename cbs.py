import numpy as np
import seaborn as sns


def cbs_stat(x, start, end):
    x0 = x[start:end] - np.mean(x[start:end])
    n = end - start
    y = np.cumsum(x0)

    e0 = np.argmin(y)
    e1 = np.argmax(y)

    i0 = min(e0,e1)
    i1 = max(e0,e1)
    s0 = y[i0]
    s1 = y[i1]
    return (s1-s0)**2*n/(i1-i0)/(n-i1+i0), i0+start, i1+start



def cbs(x, shuffles=100):
    start=0
    end=len(x)
    max_t, max_start, max_end = cbs_stat(x ,start,end)
    thresh_count=0
    for i in range(shuffles):
        threshold = cbs_stat(np.random.permutation(x[start:end]),0,end-start)[0]
        if threshold >= max_t:
            thresh_count+=1
        if thresh_count > 5:
            return False, max_t, max_start, max_end
    return True, max_t, max_start, max_end


def segment(x, start, end, L):
    threshold, t, s, e = cbs(x[start:end])
    print(start, end, threshold, t, start+s, start+e)
    if not threshold:
        L.append((start,end))
    else:
        if s>1:
            segment(x,start, start+s,L)
        if e-s>1:
            segment(x,start+s, start+e,L)
        if start+e < end-1:
            segment(x,start+e,end,L)
        return L
    

if __name__ == '__main__':
    N = 20
    means = np.random.randint(-5, 5, N)
    runs = np.random.randint(8, 15, N)
    sample = np.concatenate([([x]*y)+np.random.normal(0,.5,size=y) for (x,y) in zip(means,runs)])
    sample = sample 

    L=[]
    segment(sample, 0, len(sample), L)
    j=sns.scatterplot(list(range(len(sample))),sample)
    for x in L:
        j.axvline(x[0])
    j.axvline(L[-1][1])
    j.axvline(0)
    j.axvline(np.cumsum(runs)[-1])
    x0=0
    
    for i,x in enumerate(np.cumsum(runs)):
        j.hlines(means[i],x0, x ,color='orange',linewidth=2.0)
        x0 = x
    f = j.get_figure()
    f.set_size_inches(8,8)
    f.savefig('plot.png')
