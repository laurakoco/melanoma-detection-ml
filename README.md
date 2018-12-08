# Melanoma Detection via Convolutional Neural Network (CNN)

The objective of this project is to create a Convolutional Neural Network (CNN) to classify a dermoscopic image of a skin lesion as Melanoma or Non-Melanoma. A dermoscopic image is a picture of the skin using a microscope and illumination.

## Motivation

Melanoma is the deadliest and most aggressive form of skin cancer; it is projected that in 2018, Melanoma of the skin will cause 9,320 deaths in the United States. However, if Melanoma is caught in an early stage, the 5-year survival rate is about 99%. Therefore, the early detection of Melanoma, before metastasis, is critical for patient survival.

Melanoma evolves from the rapid growth of melanin-producing cells, Melanocytes, which are located in the skin’s epidermis. Although Melanoma can only be confirmed with a biopsy, it is often identified visually in an existing or new nevus (commonly known as "mole") using the mnemonic “ABCDEs”:

1.    _Asymmetry_ – The lesion is irregular, or not symmetrical, in shape.
2.    _Border_ – The edges are irregular and difficult to define.
3.    _Color_ – More than one color, or uneven distribution of color, exists.
4.    _Diameter_ – Diameter is greater than 6 mm.
5.    _Evolving_ – The lesion has changed in color and size over time.

## Built With

* TensorFlow
* Keras
* Python
* MATLAB's Deep Learning Toolbox

## Models

Two CNN architectures were explored for this project:

1) Simple CNN built from scratch with Keras, TensorFlow, and Python.
2) Deep (AlexNet-based) CNN built with MATLAB's Deep Learning Toolbox. Final 3 layers (Fully Connected, Softmax, and Classification Output) are adapted to my dataset.

The source code for both models may be found in /src.

The block diagram of the Keras model may be seen below.

<img src="images/Keras_Block.png" width="600">

The AlexNet CNN architecture may be seen below. AlexNet is a popular CNN that was trained on subsets of ImageNet database used in the ILSVRC-2010 and ILSVRC-2012 competitions. The ImageNet database has over 15 million labeled, high-resolution images belonging to 22,000 categories. AlexNet is 8 layers deep and can classify images into 1000 categories, such as keyboard, mouse, pencil, etc.

<img src="images/AlexNet.png" width="600">

AlexNet was trained on millions of images. As such, its lower layers have learned rich feature detection (such as edges, blobs, etc.) while its higher layers are more task specific (such as recognizing a keyboard). For this project, I replaced the last 3 layers of AlexNet to allow the network to learn features that are specific to my objective. This is known as Transfer Learning. The modified architecture may be seen below via MATLAB's Deep Network Designer.

<img src="images/AlexNet_MATLAB.png" width="600">

More information on implementing AlexNet in MATLAB may be found [here](https://www.mathworks.com/help/deeplearning/ref/alexnet.html).

## Data

Neural Networks are a form of supervised learning, so training, testing, and validation data is required. This dataset has been acquired from multiple public sources (PH2, ISIC, and HAM10000) and combined, yielding a multi-source dataset:

* [ISIC](https://isic-archive.com)
* [PH2](http://www.fc.up.pt/addi/ph2%20database.html)
* [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

I visually inspected the images from these sources and selected only the high-quality and representative images to add to my dataset. In addition, I manually cropped the selected images.

Altogether, my combined dataset contains 2,148 images. This dataset is randomly split into 70% for training, 20% for testing, and 10% for validation.

My multi-scource dataset can be downloaded [here](https://drive.google.com/open?id=1VFO37HNONIY_8qWC_wmhXgjvOKcg3zJU).

The details of this dataset may be seen below.

|            | **Melanoma** | **Non-Melanoma** |
|:----------:|:--------:|:--------------:|
|    Train   |    751   |      754     |
|    Test    |    214   |      215     |
| Validation |    106   |      108     |
|    **Total**   |   1,071  |     1,077    |

- Sample images of Melanoma from the dataset:

<img src="images/Melanoma.jpg" height="200"> <img src="images/Melanoma_2.jpg" height="200">

- Sample images of Non-Melanoma from the dataset:

<img src="images/Non-Melanoma.jpg" height="200"> <img src="images/Non-Melanoma_2.jpg" height="200">

### Directory Structure

This directory structure for the Keras model must be as follows. This is the structure of data_keras.zip.

data /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;melanoma /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;non-melanoma /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;melanoma /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;non-melanoma /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;validation /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;melanoma /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;non-melanoma /

The directory structure for the MATLAB AlexNet model must be as follows:

data /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Melanoma /

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Non-Melanoma /

## Results

### Keras CNN

The CNN built in Keras is able to achieve an overall accuracy of 78.8% on testing data not exposed to the CNN during training. This is actually good for such a simple CNN. This model takes about 10 minutes (12 epochs) to train on my MacBook Pro. While this CNN is simple and straightforward to understand, it does not yield the same level of accuracy as a deeper CNN. Below shows the accuracy, sensitivity, and specificty on the testing data. This model has the following parameters:

- Adam optizmier (default parameters) [1]
- Fully-connected neurons = 512
- 3 Conv2D layers
- Epochs = 12

|   Accuracy  | 78.8 |
|:-----------:|:-:|
| Sensitivity | 69.2 |
| Specificity | 88.3 |

### Deep AlexNet-Based CNN

The deep, AlexNet-based CNN is able to achieve an overall accuracy of 90.2% on testing data not exposed to the CNN during training. This takes quite a long time to run (20 epochs = 450 minutes). Below shows the accuracy, sensitivity, and specificty on the testing data. This model has the following parameters:

- Stochastic Gradient Descent with Momentum (SGDM) optimizer; momentum = 0.9
- Learning rate = 1e-5
- Epochs = 20

|   Accuracy  | 90.2 |
|:-----------:|:-:|
| Sensitivity | 89.2 |
| Specificity | 91.2 |

Below is the training data for this model.

<img src="images/AlexNet_Training.png" width="1000" align="left">
<br>
Below are AlexNet-based model predictions (and associated probability) on images on Melanoma and Non-Melanoma not exposed to the CNN during training. As you can see, this model classifies images very well with accuracy > 90%.

<img src="images/AlexNet_Classification.jpg" width="1000" align="left">

<img src="images/AlexNet_Classification_2.jpg" width="1000" align="left">

### Implications

An important question to ask ourselves is what accuracy is good enough? A small study published in 2018, dermatologists accurately diagnosed Melanoma (sensitivity) with 86.6% and 88.9% accuracy, depending on the stage of the Melanoma [2]. This CNN is able to detect Melanoma with higher accuracy than a dermatologist. It’s possible a dermatologist or other doctors not specialized in dermatology (e.g. primary care physician) may use a CNN as a tool to aid the diagnosis of Melanoma.

## Future Work

I plan to improve this accuracy further by automating image pre-processing and/or obtaining a larger dataset. I also plan to integrate this model into a website or app so that users may upload an image and see the associate risk of the mole.

## Author

**Laura Kocubinski** [laurakoco](https://github.com/laurakoco)

## Acknowledgments

* Boston University MET Master Science Computer Science Program
* MET CS 767 Machine Learning

## References

[1] Kingma, Diederik, and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” 3rd International Conference for Learning Representations, San Diego, 2015.

[2] Haenssle, H. A., C Fink, R. Schneiderbauer, F. Toberer, T. Buhl, A. Blum, A. Kalloo, A. Ben Hadj Hassen, L. Thomas A Enk and L. Uhlmann. “Man Against Machine: Diagnostic Performance of a Deep Learning Convolutional Neural Network for Dermoscopic Melanoma Recognition in Comparison to 58 Dermatologists.” Annals of Oncology, August 2018.

# melanoma_detection_ml
