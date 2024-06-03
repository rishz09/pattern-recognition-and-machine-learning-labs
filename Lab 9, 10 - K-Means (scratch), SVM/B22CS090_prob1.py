import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

class Kmeans_scratch:
    def __init__(self, n_init, max_iter, k, random_seed, tol=1e-4):
        #number of times cluster centers will be randomly initialised
        self.n_init = n_init
        #max number of iterations the cluster centers will be updated for
        self.max_iter = max_iter
        #number of clusters
        self.k = k
        #maximum accepted tolerance for two cluster centers to be considered equal
        self.tol = tol
        self.random_seed = random_seed
        #storing the lowest inertia value at the end
        self.inertia = None

        np.random.seed(self.random_seed)
        self.seed_list = np.random.randint(low=0, high=100000, size=self.n_init)


    #computing cluster center
    def computeCentroid(self, features):
        return np.mean(features, axis=0)
    
    
    #calculating inertia to compare best initialisation run
    def calc_inertia(self, dist):
        inertia = np.sum(np.min(dist, axis=1), dtype=np.float64)
        return inertia / dist.shape[0]
    

    #calculating distance between every pixel and every cluster center
    def calc_distance(self, x, cluster_centers):
        rows = x.shape[0]
        dist = np.empty([rows, self.k])
        dist = np.sqrt(np.sum((x[:, np.newaxis, :] - cluster_centers) ** 2, axis=2))

        return dist


    #updating cluster centers by categorically (label wise) finding out new cluster centers
    def update_cluster_centers(self, x, cluster_labels):
        updated_cluster = np.empty((self.k, x.shape[1]))

        for i in range(self.k):
            cluster_i_points = x[np.where(cluster_labels == i)[0]]
            updated_cluster[i, :] = self.computeCentroid(cluster_i_points)

        return updated_cluster


    #function which trains the model and eventually returns the best cluster centers
    #and labels with respect to the lowest inertia
    def mykmeans(self, x):
        cluster_centers_final = None
        set_cluster_idx_final = None

        #initialising cluster centers for n_init times
        for i in range(self.n_init):
            print(f'Init {i+1}')
            cluster_centers = np.empty((self.k, x.shape[1]))

            np.random.seed(self.seed_list[i])
            #choosing random cluster center for the first time in a run
            random_idx = np.random.randint(low=0, high=x.shape[0], size=self.k)
            cluster_centers = x[random_idx]
            set_cluster_idx = None
            temp_inertia = None

            for i in range(self.max_iter):
                #keeping old cluster centers for comparison
                old_cluster_centers = cluster_centers.copy()
                dist = self.calc_distance(x, cluster_centers)
                temp_inertia = self.calc_inertia(dist)
                
                #finding cluster center which is closest to every pixel
                set_cluster_idx = np.argmin(dist, axis=1)

                #updating new cluster center
                cluster_centers = self.update_cluster_centers(x, set_cluster_idx)

                #implementing converging criteria
                if np.linalg.norm(cluster_centers - old_cluster_centers) < self.tol:
                    print(f'Stopped at Iteration {i+1}')
                    break
            
            #updating inertia, cluster centers and labels if we have got lower
            #inertia value
            if (self.inertia == None) or (temp_inertia < self.inertia):
                self.inertia = temp_inertia
                cluster_centers_final = cluster_centers
                set_cluster_idx_final = set_cluster_idx

        return cluster_centers_final, set_cluster_idx_final
    


class Kmeans_spatial_coherence:
    def __init__(self, n_init, max_iter, k, rows, cols, random_seed, tol=1e-4):
        self.n_init = n_init
        self.max_iter = max_iter
        self.k = k
        #storing number of rows and columns in original image to assign
        #coordinates to every pixel
        self.r = rows
        self.c = cols
        self.tol = tol
        self.coord = None
        self.inertia = None
        self.random_seed = random_seed

        np.random.seed(self.random_seed)
        self.seed_list = np.random.randint(low=0, high=100000, size=self.n_init)

    #filling a 2D array with its corresponding 2D coordinates and then flattening it    
    def fill_coordinates(self):
        self.coord = np.empty((self.r, self.c, 2), dtype=np.float64)

        for i in range(self.r):
            for j in range(self.c):
                #min - max normalisation of coordinates
                self.coord[i][j][0] = (i / self.r)
                
                #assigning importance to coordinates for calculating distance
                #find more about this on the report
                self.coord[i][j][0] = self.coord[i][j][0] * np.sqrt(0.25)

                self.coord[i][j][1] = (j / self.c)
                self.coord[i][j][1] = self.coord[i][j][1] * np.sqrt(0.25)

        self.coord = self.coord.reshape(-1, 2)


    #stacking the flattened image and the coordinates in order to attain 
    #a matrix with 5 features (3 colours RGB + 2 coordinates X and Y)
    def stack_coords(self, x):
        stacked_arr = np.concatenate((x, self.coord[:, 0].reshape((-1, 1)), self.coord[:, 1].reshape((-1, 1))), axis=1)  
        return stacked_arr
    

    def computeCentroid(self, features):
        return np.mean(features, axis=0)
    
    
    def calc_inertia(self, dist):
        inertia = np.sum(np.min(dist, axis=1), dtype=np.float64)
        return inertia / dist.shape[0]
    

    #calculating euclidean distance by taking the 5 features
    def calc_distance(self, x, cluster_features):
        rows = x.shape[0]
        dist = np.empty([rows, self.k])
        dist = np.sqrt(np.sum((x[:, np.newaxis, :] - cluster_features) ** 2, axis=2))

        return dist


    def update_cluster_centers(self, x, cluster_labels):
        updated_cluster_feature = np.empty((self.k, x.shape[1]))

        for i in range(self.k):
            cluster_i_points = x[np.where(cluster_labels == i)[0]]
            updated_cluster_feature[i, :] = self.computeCentroid(cluster_i_points)

        return updated_cluster_feature


    def mykmeans(self, x):
        cluster_features_final = None
        set_cluster_idx_final = None

        self.fill_coordinates()
        
        #assinging feature importance to the colours (find more in report)
        x = x * np.sqrt(0.75)

        #updating x with a total of 5 features
        x = self.stack_coords(x)

        #the following contains same implementation as that of Kmeans_scratch class
        for i in range(self.n_init):
            print(f'Init {i+1}')

            np.random.seed(self.seed_list[i])
            random_idx = np.random.randint(low=0, high=x.shape[0], size=self.k)
            cluster_features = x[random_idx]
            set_cluster_idx = None
            temp_inertia = None

            for i in range(self.max_iter):
                old_cluster_features = cluster_features.copy()
                dist = self.calc_distance(x, cluster_features)
                temp_inertia = self.calc_inertia(dist)

                set_cluster_idx = np.argmin(dist, axis=1)
                cluster_features = self.update_cluster_centers(x, set_cluster_idx)

                #implement converging criteria over here
                if np.linalg.norm(cluster_features - old_cluster_features) < self.tol:
                    print(f'Stopped at Iteration {i+1}')
                    break
            
            if (self.inertia == None) or (temp_inertia < self.inertia):
                self.inertia = temp_inertia
                cluster_features_final = cluster_features
                set_cluster_idx_final = set_cluster_idx


        return cluster_features_final, set_cluster_idx_final
    

def main():
    img = Image.open('test.png')
    img_arr = np.array(img)
    plt.figure()
    plt.imshow(img_arr)
    plt.show(block=False)

    img_reshape = img_arr.reshape(-1, 3)
    #normalising pixel values to min = 0 and max = 1
    img_reshape_norm = img_reshape / 255

    k_vals = [2, 4, 6, 8, 10]

    for val in k_vals:
        kmeans = Kmeans_scratch(n_init=5,
                                max_iter=300,
                                k=val,
                                random_seed=123)

        cluster_centers, cluster_labels = kmeans.mykmeans(img_reshape_norm)
        print((cluster_centers * 255).astype(np.uint8))

        #getting segmented image with k colours only
        new_img = cluster_centers[cluster_labels]
        new_img = new_img.reshape(img_arr.shape)
        new_img = new_img * 255
        new_img = new_img.astype(np.uint8)
        plt.figure()
        plt.imshow(new_img)
        plt.title(f'Kmeans from scratch, k = {val}')
        plt.axis('off')
        plt.savefig(f'images/kmeans_scratch_k{val}.png')
        plt.show(block=False)


    #using KMeans from sklearn
    for val in k_vals:
        kmeans = KMeans(n_clusters=val, max_iter=300, n_init=5, random_state=123)
        kmeans.fit(img_reshape_norm)

        print((kmeans.cluster_centers_ * 255).astype(np.uint8))
        img_from_sklearn = kmeans.cluster_centers_[kmeans.labels_]
        img_from_sklearn = img_from_sklearn.reshape(img_arr.shape)
        img_from_sklearn = img_from_sklearn * 255
        img_from_sklearn = img_from_sklearn.astype(np.uint8)
        
        plt.figure()
        plt.imshow(img_from_sklearn)
        plt.title(f'Kmeans from sklearn, k = {val}')
        plt.axis('off')
        plt.savefig(f'images/kmeans_sklearn_k{val}.png')
        plt.show(block=False)

    
    #using kmeans with spatial coherence
    kmeans = Kmeans_spatial_coherence(n_init=5,
                                    max_iter=300,
                                    k=8,
                                    rows=512,
                                    cols=512,
                                    random_seed=123,
                                    tol=1e-4)

    cluster_features, labels = kmeans.mykmeans(img_reshape_norm)

    new_img = cluster_features[labels]
    new_img = new_img[:, :3]

    new_img = new_img.reshape(img_arr.shape)
    new_img = new_img * 255
    new_img = new_img.astype(np.uint8)
    plt.figure()
    plt.imshow(new_img)
    plt.title(f'Kmeans Spatial Coherence (Balanced importance), k = 8')
    plt.axis('off')
    plt.savefig(f'images/kmeans_spatial_coherence_balanced_k8.png')
    plt.show()



if __name__ == '__main__':
    main()