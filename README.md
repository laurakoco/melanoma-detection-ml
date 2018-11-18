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
* MATLAB

## Data

Since Neural Networks are a form of supervised learning, training, testing, and validation images are required. These images have been acquired from multiples sources (PH2, ISIC, and HAM10000) and combined, yielding a multi-sourced dataset:

* [ISIC](https://isic-archive.com)
* [PH2](http://www.fc.up.pt/addi/ph2%20database.html)
* [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

- Sample image of Melanoma:

![](images/Melanoma.jpg?raw=true)

- Sample image of Non-Melanoma:

![](images/Non-Melanoma.jpg?raw=true)

I have a combined dataset of 2,148 images. This dataset is divided into subsets for testing, training, and validation: 70%, 20%, and 10%, respectively. The details of the subsets may be seen below.

|            | **Melanoma** | **Non-Melanoma** |
|:----------:|:--------:|:--------------:|
|    Train   |    751   |      754     |
|    Test    |    214   |      215     |
| Validation |    106   |      108     |
|    **Total**   |   1,071  |     1,077    |

My dataset can be found [here](link to dataset)

This directory is stored locally on my machine with the following structure:

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


## Results

The CNN built in Keras is able to achieve an overall accuracy of X % on testing data not exposed to the CNN during training:

|:-----------:|:-:|
|   Accuracy  | X |
| Sensitivity | X |
| Specificity | X |


## Author

**Laura Kocubinski** [laurakoco](https://github.com/laurakoco)

## Acknowledgments

* Boston University MET Master Science Computer Science Program
* MET CS 767 Machine Learning

# melanoma_detection_ml
