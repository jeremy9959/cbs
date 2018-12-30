# cbs
A simple implementation of the circular binary segmentation algorithm in python.

```python
import cbs
data = cbs.generate_normal_time_series(10)
L = segment(data)
```

The resulting list L contains pairs (x,y) where each slice data[x:y] is a distinct segment of the data.

