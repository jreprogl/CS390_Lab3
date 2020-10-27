import pdb
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
#from scipy.misc import imresize#imsave, imresize
from imageio import imwrite as imsave
from cv2 import resize as imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
from PIL import Image
 
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)
 
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
CONTENT_IMG_PATH = "/content/drive/My Drive/CS390/mando500.jpg"           #TODO: Add this.
STYLE_IMG_PATH = "/content/drive/My Drive/CS390/starry500.jpg"             #TODO: Add this.
 
 
CONTENT_IMG_H = 500
CONTENT_IMG_W =500
 
STYLE_IMG_H = 500
STYLE_IMG_W = 500
 
CONTENT_WEIGHT = .01    # Alpha weight.
STYLE_WEIGHT = 1   # Beta weight.
TOTAL_WEIGHT = 0.05
 
TRANSFER_ROUNDS = 200
 
loss_grads=0
new_grads = 0
new_loss = 0
 
#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img
 
 
#def gramMatrix(x):
#    x = tf.transpose(x, (2, 0, 1))
#    features = tf.reshape(x, (tf.shape(x)[0], -1))
#    gram = tf.matmul(features, tf.transpose(features))
#    return gram
 
def gramMatrix(x):
    if K.image_data_format() == "channels_first":
        features = K.flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
 
 
 
#========================<Loss Function Builder Functions>======================
 
# Need to determine numFilters
def styleLoss(style, gen):
    numFilters = 3
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen)) / (4. * (numFilters**2) * ((CONTENT_IMG_H*CONTENT_IMG_W)**2)))
    #return None   #TODO: implement.
 
 
def contentLoss(content, gen):
    return K.sum(K.square(gen - content))
 
 
def totalLoss(x):
    loss = x * TOTAL_WEIGHT
    return loss   #TODO: implement.
 
 
 
 
 
#=========================<Pipeline Functions>==================================
 
def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))
 
 
 
def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #img = imresize(img, (ih, iw, 3))
        img = np.resize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img
 
 
def computeLoss(contentTensor, styleTensor, genTensor):
    print("   Building transfer model.")
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)   #TODO: implement.
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = K.variable(0.)
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2" 
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss = loss + CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)   #TODO: implement.
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        loss = loss + (STYLE_WEIGHT/len(styleLayerNames)) * styleLoss(styleOutput, genOutput)   #TODO: implement.
        #loss = loss + STYLE_WEIGHT * styleLoss(styleOutput, genOutput)
    loss = loss + totalLoss(loss)   #TODO: implement.
    return loss
 
 




def k_function_loss(x):
  x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
  outputs = loss_grads([x])
  loss_value = outputs[0]
  grad_values = outputs[1].flatten().astype('float64')
  global new_loss
  global new_grads
  new_loss = loss_value
  new_grads = grad_values
  return loss_value

def k_function_grads(X):
  global new_grads
  grad_values = np.copy(new_grads)
  return grad_values
 
 
 
 
'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # TODO: Setup gradients or use K.gradients().
   # pdb.set_trace()
    loss = computeLoss(contentTensor, styleTensor, genTensor)
 
    #tf.compat.v1.disable_eager_execution()
    grads = K.gradients(loss, genTensor)[0]
    outputs = [loss]
    outputs.append(grads)
    global loss_grads 
    loss_grads = K.function([genTensor], outputs)
    print("   Beginning transfer.")
    tData = tData.flatten()
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        tData, min_val, info = fmin_l_bfgs_b(k_function_loss, tData, fprime=k_function_grads, maxfun=40)
        print("      Loss: %f." % min_val)
 #       print(tData)
 #       print(tData.shape)
        img = np.copy(tData)
#        print(img.shape)
        img = np.reshape(img, (CONTENT_IMG_H, CONTENT_IMG_W, 3))
#        print(img.shape)
        img = deprocessImage(img)
        saveFile = "/content/drive/My Drive/CS390/SmallMando/testImage" + str(i) + ".png"   #TODO: Implement.
        #imsave(saveFile, img)   #Uncomment when everything is working right.
        print(img.shape)
        img = Image.fromarray(img, 'RGB')
        img.save(saveFile)
        print("      Image saved to \"%s\"." % saveFile)
    print(np.array_equal(tData, prevTData))
    print("   Transfer complete.")
 
 
 
 
 
#=========================<Main>================================================
 
def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")
 
 
 
if __name__ == "__main__":
    main()
