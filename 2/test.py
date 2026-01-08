import numpy as np
a=np.array([1,2,2,2,3,2,2,2,4])
count = np.bincount(a,minlength=10)
print(dict(enumerate(count)))