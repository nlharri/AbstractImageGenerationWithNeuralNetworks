import os
import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from skimage.color import hsv2rgb
from PIL import Image

#tf.keras.backend.set_learning_phase(1)
#init = tf.global_variables_initializer() # <<<<<<<<< 1
#init = initializers.VarianceScaling(scale=variance)

# --------------------------------------------
# define default constants for generation
# --------------------------------------------
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
SCALE = 1.0
LAYER_WIDTH = 32
NEURAL_NETWORK_DEPTH = 6
RANDOM_SEED = 123
VARIANCE = 60

# --------------------------------------------
# set random seeds
# --------------------------------------------
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------
# create a random number of normal distribtion:
#  - mean = 0
#  - standard deviation = 1
#  - generate only 1 number (this is the third parameter)
# https://www.sharpsightlabs.com/blog/numpy-random-normal/
# --------------------------------------------
#rnd_normal = np.random.normal(0,1,1)

# --------------------------------------------
# generate grids
# --------------------------------------------
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

# --------------------------------------------
# build the model
# --------------------------------------------
init = tf.keras.initializers.VarianceScaling(scale=VARIANCE)

input_layer = output_layer = tf.keras.layers.Input(shape=(2,))

for _ in range(0, NEURAL_NETWORK_DEPTH):
    output_layer = tf.keras.layers.Dense(
        LAYER_WIDTH,
        kernel_initializer=init,
        activation='tanh')(output_layer)

output_layer = tf.keras.layers.Dense(
    1,
    activation='tanh')(output_layer)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='rmsprop', loss='mse')

# --------------------------------------------
# generate and save the images
# --------------------------------------------
concatenated_params = np.concatenate(np.array((x,y)), axis=1)

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


# n = min(len(img), 9)
# rows = int(math.sqrt(n))
# cols = n // rows
# fig = plt.figure()
# for i in range(1, n+1):
#     image = img[i-1]
#     fig.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.imshow(image)
# plt.show()


if not os.path.exists("./results"): os.makedirs("./results")
print("results are saved to: {}".format("./results"))

timestr = time.strftime("%Y%m%d-%H%M%S")
suffix = f".var{VARIANCE:.0f}.seed{RANDOM_SEED}"
image_name = f"img.{timestr}{suffix}.png"
image_path = os.path.join("./results", image_name)
file = Image.fromarray(img)
file.save(image_path)
