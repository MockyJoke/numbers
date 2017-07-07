
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.metrics import accuracy_score
from skimage import color
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm


# In[2]:

def my_rgb2lab(colors): 
    old_shape = colors.shape
    reshaped = colors.reshape(old_shape[0],1,old_shape[1])
    lab = color.rgb2lab(reshaped)
    return lab.reshape(old_shape)


# In[3]:

OUTPUT_TEMPLATE = (
    'Bayesian classifier: {bayes_rgb:.3g} {bayes_lab:.3g}\n'
    'kNN classifier:      {knn_rgb:.3g} {knn_lab:.3g}\n'
    'SVM classifier:      {svm_rgb:.3g} {svm_lab:.3g}\n'
)


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 114, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=71, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.imshow(pixels)


    
def main():
    data = pd.read_csv(sys.argv[1])
    X = data # array with shape (n, 3). Divide by 255
    y = data # array with shape (n,) of colour words

    
    rgb_columns = ["R","G","B"]
    data[rgb_columns] = data[rgb_columns].values/255
    X_train,X_test,y_train,y_test = model_selection.train_test_split(data[rgb_columns].values,data["Label"].values)
    
    bayes_rgb_model = GaussianNB()

    bayes_lab_model = pipeline.make_pipeline(preprocessing.FunctionTransformer(my_rgb2lab),GaussianNB())

    knn_rgb_model = KNeighborsClassifier(15)
    
    knn_lab_model = pipeline.make_pipeline(preprocessing.FunctionTransformer(my_rgb2lab),KNeighborsClassifier(15))
    
    svc_rgb_model = svm.SVC(C=30)
    
    svc_lab_model = pipeline.make_pipeline(preprocessing.FunctionTransformer(my_rgb2lab),svm.SVC(C=1.0,kernel="linear", decision_function_shape="ovr"))
    
    # train each model and output image of predictions
    models = [bayes_rgb_model, bayes_lab_model, knn_rgb_model, knn_lab_model, svc_rgb_model, svc_lab_model]
    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)
        plot_predictions(m)
        plt.savefig('predictions-%i.png' % (i,))

    print(OUTPUT_TEMPLATE.format(
        bayes_rgb=bayes_rgb_model.score(X_test, y_test),
        bayes_lab=bayes_lab_model.score(X_test, y_test),
        knn_rgb=knn_rgb_model.score(X_test, y_test),
        knn_lab=knn_lab_model.score(X_test, y_test),
        svm_rgb=svc_rgb_model.score(X_test, y_test),
        svm_lab=svc_lab_model.score(X_test, y_test),
    ))


if __name__ == '__main__':
    main()

