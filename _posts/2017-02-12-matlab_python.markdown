| asdf |      matlab     |        python        |
| ---- | --------------- | -------------------- |
|      | rand(m, n)      | np.random.rand(m, n) |
|      | size(A, 1)      | A.shape[0]           |
|      | sum(A, 1)       | A.sum(axis = 0)      |
|      | repmat(A, m, n) | np.tile(A, (n, m)).T |
|      | numel(A)        | A.size               |
|      | vec(A), A(:)    | A.flatten(1)         |
|      | ones(m,n)       | np.ones((m, n))      |
|      | zeros(m, n)     | np.zeros((m, n))     |
|      | eye(m)          | np.eye(m)            |
|      | max(A, 1)       | np.amax(A, axis = 0) |
|      | max(A(:))       | np.amax(A)           |
|      | min(A, 1)       | np.amin(A, axis = 0) |
|      |                 |                      |
