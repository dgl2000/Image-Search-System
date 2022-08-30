# README

@file: This is the coursework description for Computer Vision. Image search is the topic for this project.

@author: [Gaole Dai ](mailto:gd25@rice.edu)  :penguin:

<img src="https://drive.google.com/uc?export=view&id=1Rxurvdt3aaCSA2O4H2qr7pj5cCETKamC" alt="index page" style="zoom: 40%;" />

## File Structure

* README
* src
  * Source code for the image search algorithm implementation, which includes the SIFT and VLAD method, ResNet18 model method, Autoencoder method.

* website
  * Code for the web based application. Run the app.py in the `website/api` file to view the web application. The demo video is here: https://drive.google.com/file/d/1_DivjrptO42tvEio7E7tHXZqN87uXQy5/view?usp=sharing

* models
  * It should be included, but due to the Moodle file size limitation, I did not include the models in this zip, if you have any doubts, feel free to contact me! Or you could download them via the link below:
    * https://drive.google.com/drive/folders/1sIfIDF9eo7ltGgABH2AbfGAHDjjw3Dbe?usp=sharing
* To properly run the web-based application, the images should be downloaded from the link below and place in the `website/img` folder, and the root folder.
  * https://drive.google.com/drive/folders/1L8Qh2Kt3hCANELPWdkveatBhgEKvypjo?usp=sharing


## Dependence and Environment

The model training and feature extractor are executed in Colab with GPU, the web application is developed with PyCharm.

1. torch == 1.11.0
2. tqdm == 4.64.0
3. flask == 1.1.2
4. py-opencv == 4.5.5
5. numpy ==1.20.0
6. pandas == 1.3.4
7. scikit-learn == 1.0.2

## Citation

background image for index page: https://www.britain-magazine.com/oxford/

test image: https://en.wikipedia.org/wiki/Radcliffe_Camera