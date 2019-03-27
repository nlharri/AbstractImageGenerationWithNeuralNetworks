import os
import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from skimage.color import hsv2rgb
from PIL import Image

import imageio

# --------------------------------------------
# define default constants for generation
# --------------------------------------------
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
SCALE = 1.0
RANDOM_SEED = 7894
MAXVARIANCESTEPS = 1

# --------------------------------------------
# generate grid
# --------------------------------------------
def generate_grid():
    mean = np.mean((IMAGE_WIDTH, IMAGE_HEIGHT))
    x = np.linspace(
        -IMAGE_WIDTH / mean * SCALE,
         IMAGE_WIDTH / mean * SCALE,
         IMAGE_WIDTH)

    y = np.linspace(
        -IMAGE_HEIGHT / mean * SCALE,
         IMAGE_HEIGHT / mean * SCALE,
         IMAGE_HEIGHT)

    X, Y = np.meshgrid(x, y)

    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)

    return x, y

# --------------------------------------------
# build the model
# --------------------------------------------
def build_model(number_of_input_params, variance, black_and_white = False, layer_width = 32, neural_network_depth = 8, activation=tf.tanh):

    # --------------------------------------------
    # set random seeds
    # --------------------------------------------
    tf.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    init = tf.keras.initializers.VarianceScaling(scale=variance)

    input_layer = output_layer = tf.keras.layers.Input(
        shape=(number_of_input_params,))

    for _ in range(0, neural_network_depth):
        output_layer = tf.keras.layers.Dense(
            layer_width,
            kernel_initializer=init,
            activation='tanh')(output_layer)

    output_layer = tf.keras.layers.Dense(
        1 if black_and_white else 3,
        kernel_initializer=init,
        activation='tanh')(output_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='mse')

    return model


# --------------------------------------------
# generate and save the images
# --------------------------------------------
if __name__ == "__main__":

    x, y = generate_grid()
    concatenated_params = np.concatenate(np.array((x,y)), axis=1)

    filenames = []
    for variancestep in range(0, MAXVARIANCESTEPS):
        #variance = (variancestep+1) * 0.09 + 2.5
        variance = (variancestep+11) * 0.9 + 2.5

        model = build_model(
            number_of_input_params=2, 
            variance=variance, 
            black_and_white = False, 
            layer_width = 32)
        
        pred = model.predict(concatenated_params)

        img = []
        channels = pred.shape[1] # can be 1 (greyscale) or 3 (hsv)
        for channel in range(channels):
            yp = pred[:, channel]
            yp = (yp - yp.min()) / (yp.max()-yp.min())
            img.append(yp.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
        img = np.dstack(img)

        if channels == 3: img = hsv2rgb(img)
        img = (img * 255).astype(np.uint8)
        img = img.squeeze()

        if not os.path.exists("./results"): os.makedirs("./results")

        timestr = time.strftime("%Y%m%d-%H%M%S")
        suffix = ".var{:.4f}.seed{}".format(variance, RANDOM_SEED)
        image_name = "img.{}{}.png".format(timestr, suffix)
        image_path = os.path.join("./results", image_name)
        print("results are saved to: {}".format(image_path))
        file = Image.fromarray(img)
        file.save(image_path)
        filenames.append(image_path)


    with imageio.get_writer('./results/movie.gif', mode='I', duration=0.25) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
