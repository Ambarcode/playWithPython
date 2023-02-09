# playWithPython

The Repo is for Machine Learning lovers....

Machine Learning is said as a subset of artificial intelligence that is mainly concerned with the development of algorithms which allow a computer to learn from the data and past experiences on their own


-K means Clustering(On Iris DataSet)

K-means is a popular clustering algorithm that aims to partition a set of points into K clusters, where each cluster is represented by its centroid. In this documentation, we will walk through the steps of applying K-means clustering on the Iris dataset. The Iris dataset consists of 150 instances and 4 features (sepal length, sepal width, petal length, petal width).

Step 1: Load the Iris dataset
We start by loading the Iris dataset into our Python environment. This can be done using the sci-kit learn library.

from sklearn import datasets
iris = datasets.load_iris()

Step 2: Preprocessing the data
Before applying K-means, we need to preprocess the data by normalizing it. Normalization helps to ensure that each feature contributes equally to the clustering process.

from sklearn import preprocessing
x = iris.data
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)

Step 3: Apply K-means
We use the KMeans class from sci-kit learn to apply K-means on the Iris dataset. We set the number of clusters to 3, which is the number of target classes in the Iris dataset.

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(x_scaled)

Step 4: Evaluate the Clustering
To evaluate the clustering, we can use the inertia_ attribute, which returns the sum of squared distances of samples to their closest cluster center. A lower inertia value indicates a better clustering result.

print("Inertia: ", kmeans.inertia_)

Step 5: Visualize the Clustering
We can use a scatter plot to visualize the clustering result. The scatter plot shows the 150 instances and color-codes each instance according to its cluster assignment.


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=kmeans.labels_)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
