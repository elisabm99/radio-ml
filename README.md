# radio-ml
Code for the preparation of the images to train the neural network of 2306.15720

Download T-Recs catalogs from 1805.05222 to the catalogs/ folder.
Run prepare_catalog_and_labels.py with arguments --alpha --S0.
These two parameters define the differential number count of sources as a function of flux to include in the image:
![equation](https://latex.codecogs.com/png.image?\dpi{110}n(s)=\left(1&plus;\frac{s_0}{s}\right)^\alpha&space;n_{\rm&space;TRECS}(s))

The code creates a catalog of sources to be fed to Yandasoft in order to create an image. It also creates the labels for the CNN training.