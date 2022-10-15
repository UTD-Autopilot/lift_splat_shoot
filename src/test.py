import numpy as np

a = np.array([[[1, 1],[1, 1]]])
b = np.array([[[2, 2],[2, 2]]])

c = np.expand_dims(a, 0) * np.expand_dims(b, 1)
print(c)

print(np.expand_dims(a, 0).shape)
print(np.expand_dims(b, 1).shape)
print(c.shape)