# Abstract Image Generation Prototype with Neural Networks, using Tensorflow and Keras

Based on https://github.com/janhuenermann/blog/blob/master/abstract-art-with-ml/script.py

## Prerequisites

```pip install --upgrade tensorflow ipykernel numpy jupyter matplotlib tensorflow_datasets scikit-image imageio Pillow tqdm plaidml-keras plaidbench```

## Additional mayavi installation

```
pip install -U vtk
pip install -U ipython
pip install -U mayavi
pip install -U PyQt5
jupyter nbextension install --py mayavi --user
jupyter nbextension enable --py mayavi --user
```

## TODO

### Understand the following

- [x] numpy meshgrid
- [x] numpy ravel
- [x] numpy reshape
- [x] numpy array
- [x] numpy concatenate
- [ ] keras.initializers.VarianceScaling
- [ ] numpy dstack
- [ ] numpy squeeze
- [ ] numpy astype
- [ ] skimage color hsv2rgb
