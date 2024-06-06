import tensorflow as tf
import keras.preprocessing.image as process_im
from PIL import Image
# import matplotlib.pyplot as plt
from keras.applications import VGG19
import numpy as np
from keras.models import Model
from tensorflow.python.keras import models


    

# img-->load_fil---->arr--->show_im
def load_file(image_path):
    image = Image.open(image_path)
    max_dim = 256
    factor = max_dim / max(image.size)
    image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.LANCZOS)
    im_array = np.array(image)  # Directly convert image to array using numpy
    im_array = np.expand_dims(im_array, axis=0)  # Expand dimensions to create a batch of size 1
    return im_array



def img_preprocess(img_path):
    image=load_file(img_path) # Converting into image array
    img=tf.keras.applications.vgg19.preprocess_input(image)
    return img



def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3 #Input dimension must be [1, height, width, channel] or [height, width, channel]
  # perform the inverse of the preprocessing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1] # converting BGR to RGB channel

  x = np.clip(x, 0, 255).astype('uint8')
  return x

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
number_content=len(content_layers)
number_style =len(style_layers)

def get_model():
    vgg=tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')
    vgg.trainable=False
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    number_content=len(content_layers)
    number_style =len(style_layers)
    content_output=[vgg.get_layer(layer).output for layer in content_layers]
    style_output=[vgg.get_layer(layer).output for layer in style_layers]
    model_output= style_output+content_output
    return models.Model(vgg.input,model_output)


