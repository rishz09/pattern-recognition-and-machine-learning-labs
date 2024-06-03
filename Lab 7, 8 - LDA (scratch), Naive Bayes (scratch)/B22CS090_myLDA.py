import numpy as np

#Compute following terms and print them:
#1. Difference of class wise means = {m_1-m_2}
#2. Total Within-class Scatter Matrix S_W
#3. Between-class Scatter Matrix S_B
#4. The EigenVectors of matrix S_W^{-1}S_B corresponding to highest EigenValue
#5. For any input 2-D point, print its projection according to LDA.

#computes mean difference between the feature vectors of each individual class label
def ComputeMeanDiff(X):
    #separating data into feature matrix and its corresponding labels
    x, y = X[:, :2], X[:, 2]

    labels = np.unique(y)
    mean_vecs = np.empty((len(labels), x.shape[1]))

    for i in range(len(labels)):
        label_mean_vec = np.mean(x[y == labels[i]], axis=0)
        mean_vecs[i, :] = label_mean_vec

    diff = mean_vecs[0, :] - mean_vecs[1, :]
    diff = diff.reshape(1, 2)

    return diff
  

#computes within class scatter matrix by summing up individual class scatter matrices
def ComputeSW(X):
    x, y = X[:, :2], X[:, 2]
    labels = np.unique(y)
    mean_vecs = np.empty((len(labels), x.shape[1]))

    for i in range(len(labels)):
        label_mean_vec = np.mean(x[y == labels[i]], axis=0)
        mean_vecs[i, :] = label_mean_vec

    d = x.shape[1]  #number of features
    Sw = np.zeros((d, d))

    for label, mean_vec in zip(labels, mean_vecs):
        class_scatter = np.zeros((d, d))
        class_set = x[y == label]

        for row in class_set:
            row, mean_vec = row.reshape(d, 1), mean_vec.reshape(d, 1)
            class_scatter += (row - mean_vec) @ (row - mean_vec).T

        Sw += class_scatter

    return Sw


#computes between class scatter matrix
def ComputeSB(X):
  diff = ComputeMeanDiff(X)
  Sb = (diff.T).dot(diff)
  return Sb


#computes projection vector
def GetLDAProjectionVector(X):
  Sw = ComputeSW(X)
  Sb = ComputeSB(X)

  #performing eigenvalue decomposition to get eigenvalues and their corresponding eigenvectors
  eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
  sorted_indices = np.argsort(eigen_vals)[::-1]
  eigen_vals = eigen_vals[sorted_indices]
  eigen_vecs = eigen_vecs[:, sorted_indices]
  
  #projection vector is the eigenvector corresponding to the highest eigenvalue (the other eigenvalue = 0) over here
  w = eigen_vecs[:, 0].reshape(2, 1)

  return w


#for a given 2D point, returns the projected point in LDA axis
def project(x,y,w):
  arr = np.array([[x, y]])
  x_lda = arr.dot(w)

  return x_lda


#main method to input data and carry out the switch cases
def main():
  X = np.empty((0, 3))
  X = np.loadtxt('https://raw.githubusercontent.com/anandmishra22/PRML-Spring-2023/main/programmingAssignment/PA-4/data.csv', delimiter=',')

  print(X)
  print(X.shape)
  # X Contains m samples each of format (x,y) and class label 0.0 or 1.0

  opt=int(input("Input your option (1-5): "))

  match opt:
    case 1:
      meanDiff=ComputeMeanDiff(X)
      print(meanDiff)
    case 2:
      SW=ComputeSW(X)
      print(SW)
    case 3:
      SB=ComputeSB(X)
      print(SB)
    case 4:
      w=GetLDAProjectionVector(X)
      print(w)
    case 5:
      x=int(input("Input x dimension of a 2-dimensional point :"))
      y=int(input("Input y dimension of a 2-dimensional point:"))
      w=GetLDAProjectionVector(X)
      # print(f'{w}\n')
      print(project(x,y,w))


if __name__=='__main__':
   main()