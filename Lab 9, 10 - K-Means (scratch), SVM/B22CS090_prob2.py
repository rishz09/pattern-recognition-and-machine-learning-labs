import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

'''
y = -(w1 / w2) * x1 - (b/w2)  => linear decision boundary
'''
#plotting decision boundary for linear SVM
def decision_boundary_plotter(model, x, y):
    plt.figure()
    sns.scatterplot(x=x[:, 0],
                    y=x[:, 1],
                    hue=y,
                    edgecolor='black')

    w = model.coef_[0]
    b = model.intercept_[0]
    x_points = np.linspace(start=-2, stop=2, num=100)
    y_points = (-w[0] / w[1]) * x_points - (b / w[1])

    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.xlim(-2, 2)
    plt.legend(labels=['Setosa', 'Versicolor'])
    plt.plot(x_points, y_points, c='red')

def main():
    iris = datasets.load_iris(as_frame=True)
    features = iris.data
    y_vals = iris.target
    labels = iris.target_names
    labels

    df = pd.concat([features, y_vals], axis=1)

    #taking only required features
    x = df[['petal length (cm)', 'petal width (cm)', 'target']]
    x = x[(x['target'] == 0) | (x['target'] == 1)]
    x = x.reset_index(drop=True)

    y = x['target']
    x = x.drop(columns=['target'])

    x_arr = x.to_numpy()
    y_arr = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.2, stratify=y, random_state=3)

    #normalising features
    sc = StandardScaler()
    x_train_norm = sc.fit_transform(x_train)
    x_test_norm = sc.transform(x_test)

    #plotting the whole dataset
    plt.figure()
    sns.scatterplot(x=x_arr[:, 0],
                    y=x_arr[:, 1],
                    hue=y_arr,
                    edgecolor='black')

    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    # plt.legend(labels=['Setosa', 'Versicolor'])
    plt.title('Iris dataset with without virginica (unnormalised)')
    # plt.savefig('images/iris_dataset_without_virginica.png')
    plt.show(block=False)

    #training linearSVC model
    model = LinearSVC(random_state=3)
    model.fit(x_train_norm, y_train)

    decision_boundary_plotter(model, x_train_norm, y_train)
    plt.title('Decision Boundary on Training Data (Normalised)')
    # plt.savefig('images/iris_decision_boundary_training.png')
    plt.show(block=False)

    decision_boundary_plotter(model, x_test_norm, y_test)
    plt.title('Decision Boundary on Test Data (Normalised)')
    # plt.savefig('images/iris_decision_boundary_test.png')
    plt.show(block=False)


    '''
    Second section: working with make_moons() dataset
    '''
    data = datasets.make_moons(n_samples=500, shuffle=True, random_state=3, noise=0.05)

    x = data[0]
    y = data[1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=3)

    #normalising dataset
    sc = StandardScaler()
    x_train_norm = sc.fit_transform(x_train)
    x_test_norm = sc.transform(x_test)

    #plotting the training set
    x_train_class_0 = x[y == 0]
    x_train_class_1 = x[y == 1]
    plt.figure()
    plt.scatter(x_train_class_0[:, 0], x_train_class_0[:, 1], label = 'Class 0', s=10)
    plt.scatter(x_train_class_1[:, 0], x_train_class_1[:, 1], label = 'Class 1', s=10)
    plt.legend()
    # plt.savefig('images/moons_training_set.png')
    plt.show(block=False)


    kernels = ['linear', 'poly', 'rbf']

    #plotting decision boundaries for different kernels
    for val in kernels:
        clf = SVC(kernel=val, random_state=3)
        clf.fit(x_train_norm, y_train)
        print(f'Training Accuracy: {clf.score(x_train_norm, y_train)}')
        print(f'Test Accuracy: {clf.score(x_test_norm, y_test)}')
        plt.figure()
        plot_decision_regions(x_train_norm, y_train, clf)
        plt.title(f'{val} kernel (training set)')
        # plt.savefig(f'images/moons_decision_boundary_{val}_training.png')
        plt.show(block=False)

        plt.figure()
        plot_decision_regions(x_test_norm, y_test, clf)
        plt.title(f'{val} kernel (test set)')
        # plt.savefig(f'images/moons_decision_boundary_{val}_test.png')
        plt.show(block=False)


    #performing randomised search to find best hyperparameters
    clf = SVC(kernel='rbf', random_state=3)
    param_distributions = {'C': stats.uniform(0, 1),
                        'gamma': stats.uniform(0, 1)}

    rcv = RandomizedSearchCV(estimator=clf,
                            param_distributions=param_distributions,
                            refit=True,
                            n_iter=100,
                            n_jobs=-1,
                            cv=10,
                            scoring='accuracy',
                            random_state=3)

    rcv.fit(x_train_norm, y_train)
    print(rcv.best_params_)

    #plotting the best model
    best_clf = rcv.best_estimator_
    plt.figure()
    plot_decision_regions(x_train_norm, y_train, best_clf)
    plt.title(f"Randomized Search: C = {round(rcv.best_params_['C'], 3)}, gamma = {round(rcv.best_params_['gamma'], 3)}")
    # plt.savefig(f"images/rcv_c055_gamma708.png")
    plt.show()


if __name__ == '__main__':
    main()