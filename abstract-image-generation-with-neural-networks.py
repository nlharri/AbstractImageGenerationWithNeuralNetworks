import os
import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from skimage.color import hsv2rgb
from PIL import Image

import imageio

# --------------------------------------------
# set PlaidML as the Keras backend. This will enable HW acceleration.
# --------------------------------------------
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# --------------------------------------------
# define default constants for generation
# --------------------------------------------
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
SCALE = 1.0
RANDOM_SEED = 5243
VARIANCE = 7.5
NUMBER_OF_IMAGES_TO_GENERATE = 1
RESULTS_FOLDER = "./results"
BLACK_AND_WHITE = False

# --------------------------------------------
# generate grid
# --------------------------------------------
def generate_grid(mixin):
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
    #r = np.sqrt(x ** 2 + y ** 2)
    if y.any() != 0:
        r = np.add(x ** 2, np.divide(np.sin(y),y) ** 2)
    else:
        r = np.add(x ** 2, 1)

    Z = np.repeat(mixin, x.shape[0]).reshape(-1, x.shape[0])

    return x, y, Z.T, r

# --------------------------------------------
# build the model
# --------------------------------------------
def build_model(
    number_of_input_params,
    variance,
    black_and_white = False,
    layer_width = 32,
    neural_network_depth = 8,
    activation='tanh'):

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
            activation=activation)(output_layer)

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

    params = generate_grid(1.0)
    concatenated_params = np.concatenate(np.array(params), axis=1)

    model = build_model(
        number_of_input_params=len(params),
        variance=VARIANCE,
        black_and_white = BLACK_AND_WHITE,
        layer_width = 32)

    print("results are saved to {}".format(RESULTS_FOLDER))

    pbar = tqdm(total=NUMBER_OF_IMAGES_TO_GENERATE)

    filenames = []
    for current_image_index in range(0, NUMBER_OF_IMAGES_TO_GENERATE):

        params = generate_grid(
            mixin = 0.5 * current_image_index / NUMBER_OF_IMAGES_TO_GENERATE)
        concatenated_params = np.concatenate(np.array(params), axis=1)

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

        if not os.path.exists(RESULTS_FOLDER): os.makedirs(RESULTS_FOLDER)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        suffix = ".idx{}.var{:.4f}.seed{}".format(
            current_image_index,
            VARIANCE,
            RANDOM_SEED)
        image_name = "img.{}{}.png".format(timestr, suffix)
        image_path = os.path.join(RESULTS_FOLDER, image_name)
        #print("results are saved to: {}".format(image_path))
        file = Image.fromarray(img)
        file.save(image_path)
        filenames.append(image_path)

        pbar.update(1)

    pbar.close()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    with imageio.get_writer(
        "{}/movie.{}.var{:.4f}.seed{}.gif".format(
            RESULTS_FOLDER,
            timestr,
            VARIANCE,
            RANDOM_SEED),
        mode='I',
        duration=0.08) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
