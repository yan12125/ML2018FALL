import numpy
import sklearn.decomposition

points = [numpy.array(p) for p in (
    (1, 2, 3), (4, 8, 5), (3, 12, 9), (1, 8, 5), (5, 14, 2),
    (7, 4, 1), (9, 8, 9), (3, 8, 1), (11, 5, 6), (10, 11, 7),
)]

N = len(points)

mean = numpy.array((0, 0, 0))
for p in points:
    mean += p
mean = mean / N
print(mean)

covariance = numpy.zeros((3, 3))
for p in points:
    row = p - mean
    col = numpy.resize(row, (3, 1))
    covariance += col * row
covariance = covariance / N
print(covariance)

w, v = numpy.linalg.eig(covariance)
print('singular_values_')
print(w)
print('components_')
print(v)

transformed = numpy.zeros((N, 3))
for idx, p in enumerate(points):
    p1 = numpy.dot(p, v[..., 2])
    p2 = numpy.dot(p, v[..., 1])
    p3 = numpy.dot(p, v[..., 0])
    transformed[idx, ...] = (p1, p2, p3)
    p1_ = numpy.round_(p1, decimals=2)
    p2_ = numpy.round_(p2, decimals=2)
    p3_ = numpy.round_(p3, decimals=2)
    print(f'({p1_}, {p2_}, {p3_})')

arr = numpy.zeros((N, 3))
for idx, p in enumerate(points):
    arr[idx, ...] = p

print(arr)

print('mean')
print(numpy.mean(arr.T, axis=1))
print('cov')
print(numpy.cov(arr.T, bias=True))

pca = sklearn.decomposition.PCA(svd_solver='full')
pca.fit(arr)
print('explained_variance_')
print(pca.explained_variance_)
print('singular_values_')
print(pca.singular_values_)
print('components_')
print(pca.components_)
transformed2 = pca.transform(arr)
print(transformed2)

reconstruction_err = 0
reconstruction_err2 = 0
for idx, p in enumerate(points):
    reconstruction_err += numpy.linalg.norm(p - transformed[idx]) ** 2
    reconstruction_err2 += numpy.linalg.norm(p - transformed2[idx]) ** 2
print(reconstruction_err)
print(reconstruction_err2)
