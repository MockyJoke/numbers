
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
import sys
from sklearn.metrics import accuracy_score
from skimage import color
from sklearn import pipeline
from sklearn import preprocessing


# In[4]:

def my_rgb2lab(colors): 
    old_shape = colors.shape
    reshaped = colors.reshape(old_shape[0],1,old_shape[1])
    lab = color.rgb2lab(reshaped)
    return lab.reshape(old_shape)


# In[6]:

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
#     data = pd.read_csv("colour-data.csv")
    data = pd.read_csv(sys.argv[1])
    X = data # array with shape (n, 3). Divide by 255
    y = data # array with shape (n,) of colour words

    # TODO: build model_rgb to predict y from X.
    # TODO: print model_rgb's accuracy_score

    # TODO: build model_lab to predict y from X by converting to LAB colour first.
    # TODO: print model_lab's accuracy_score
    
    data = pd.read_csv("colour-data.csv")
    rgb_columns = ["R","G","B"]
    data[rgb_columns] = data[rgb_columns].values/255
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(data[rgb_columns].values,data["Label"].values)
    model_rgb = GaussianNB()
    model_rgb = model_rgb.fit(X_train, Y_train)
    Y_predicted = model_rgb.predict(X_test)
    print(accuracy_score(Y_test, Y_predicted))
    
    

    model_lab = pipeline.make_pipeline(preprocessing.FunctionTransformer(my_rgb2lab),GaussianNB())
    model_lab = model_lab.fit(X_train, Y_train)
    Y_predicted_lab = model_lab.predict(X_test)
    print(accuracy_score(Y_test, Y_predicted_lab))
    
    
    plt.savefig('predictions_rgb.png')
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')


if __name__ == '__main__':
    main()

