# <u><b>Neural Style Transfer:</b></u>
<h3> -------- Author- Amar Choudhary ----------- </h3>

## Introduction
In this project , we make or transform a new image which has features of both content and style image from both content and style image that we pass in model. 

For this we have to do fine tunning on vgg19 model and calculate both content loss and style loss.

![Model-Optimisation-Based-Neural-Style-Transfer-System-5](https://github.com/AmarBackInField/NeuralStyleTransfer-v1.0/assets/126746349/c760e16b-0ad8-4afb-a8e2-4ffc2bada5f8)

## *So we got Total Loss :*

*     Total Loss = Content Loss + Style Loss


## The process involves three key steps:

* *Content Extraction:* A content image is processed through the CNN to capture its structure and essential details.

* *Style Extraction:* A style image is passed through the same network to extract style features, including textures and colors, using Gram matrices from different layers.

* *Optimization:* Starting with a copy of the content image, an iterative optimization process adjusts this image to minimize the loss function. This loss combines content loss (difference from the original content image) and style loss (difference from the style image).

## <i>We got final Results as :</i>


![2](https://github.com/AmarBackInField/NeuralStyleTransfer-v1.0/assets/126746349/87e98a74-9f8e-4aa6-b3f5-c80bd9c0cf71)

![3](https://github.com/AmarBackInField/NeuralStyleTransfer-v1.0/assets/126746349/2c63164b-c53a-441f-ba09-def9c67e0cdd)

![7](https://github.com/AmarBackInField/NeuralStyleTransfer-v1.0/assets/126746349/3803131e-996f-4f5f-b3b1-f4d2c213a76a)

![9](https://github.com/AmarBackInField/NeuralStyleTransfer-v1.0/assets/126746349/9dc3fc70-7a7b-466f-be36-fdbc101b19c7)

 ## *All results are available in Markdown Folder*



