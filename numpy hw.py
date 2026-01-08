import numpy as np

arr1 = np.array([1,2,3,4,5,6,7,8,9])
arr2 = np.array([1,3,3,4,5,6,7,8,9])
same_ele = np.sum(arr1 == arr2)
print(same_ele)