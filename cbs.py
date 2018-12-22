import numpy as np
import seaborn as sns


def cbs_stat(x, start, end):
    N = end - start
    max_start = start
    max_end = end
    max_t = 0
    for seg_start in range(start,end):
        l1 = 0
        l2 = N
        s1 = 0
        s2 = np.sum(x[start:end]) 
        for seg_end in range(seg_start, end-1):
            s1 = s1+x[seg_end]
            s2 = s2-x[seg_end]
            l1 = l1 + 1
            l2 = l2 - 1
            t = np.abs((s1/l1 - s2/l2)/np.sqrt(1/l1 + 1/l2))
            if t > max_t:
                max_t, max_start, max_end  = t, seg_start, seg_end+1
    return max_t, max_start, max_end


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
        if s>0:
            segment(x,start, start+s,L)
        if e-s>1:
            segment(x,start+s, start+e,L)
        if start+e < end:
            segment(x,start+e,end,L)
        return L
    

if __name__ == '__main__':
    N = 20
    means = np.random.randint(-5, 5, N)
    runs = np.random.randint(8, 15, N)
    sample = np.concatenate([([x]*y)+np.random.normal(0,.2,size=y) for (x,y) in zip(means,runs)])
    base_line = np.concatenate([[x]*y for (x,y) in zip(means,runs)])
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
