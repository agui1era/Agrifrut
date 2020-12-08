# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using TF Lite to classify a given image using an Edge TPU.

   To run this code, you must attach an Edge TPU attached to the host and
   install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
   device setup instructions, see g.co/coral/setup.

   Example usage (use `install_requirements.sh` to get these files):
   ```
   python3 classify_image.py \
     --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
     --labels models/inat_bird_labels.txt \
     --input images/parrot.jpg
   ```
"""

import argparse
import time
import cv2  
from PIL import Image
import classify
import tflite_runtime.interpreter as tflite
import platform
import requests # to get image from the web
import shutil # to save it locally
import json
import matplotlib.pyplot as plt 
from PIL import Image, ImageFilter 

img_input='input.jpg'
modelo_calibre="model_calibre.tflite"
etiquetas_calibre='label_calibre.txt'

model='model.tflite'
labels='labels.txt'
img_input='input.jpg'
img_output='output.jpg'

img_input2='input2.jpg'
img_output2='output2.jpg'

# Set up the image URL and filename
image_url = "http://192.168.1.10:8080/?action=snapshot"
image_url2 = "http://192.168.1.20:8080/?action=snapshot"

filename =  img_input
filename2 =  img_input2

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def main():
 
  labels = load_labels(etiquetas_calibre) if etiquetas_calibre else {}
  interpreter = make_interpreter(modelo_calibre)
  interpreter.allocate_tensors()
  while cv2.waitKey(1) & 0xFF != ord('q'):

    print()
    print('____________________________________')
    print()

    # Open the url image, set stream to True, this will return the stream content.
    try: 
      r = requests.get(image_url, stream = True)
      r2 = requests.get(image_url2, stream = True)
    except:
      print("Error al descagar imagenes")    

    # Check if the image was retrieved successfully
    if (r.status_code == 200) and (r2.status_code == 200) :
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        r2.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded1 : ',filename)
        print()

        # Open a local file with wb ( write binary ) permission.
        with open(filename2,'wb') as f2:
            shutil.copyfileobj(r2.raw, f2)
            
        print('Image sucessfully Downloaded2 : ',filename2)
        print()


    else:
        print('Images Couldn\'t be retreived')
    
    im = Image.open(filename)
    im = im.crop((50, 100, 500, 300))  
    im.save(filename)


    im2 = Image.open(filename2)
    im2 = im2.crop((150, 100, 500, 300))
    im2.save(filename2)
   
     

    # Opening the image (R prefixed to string 
    # in order to deal with '\' in paths) 
    image = Image.open(filename) 
      
    # Converting the image to greyscale, as edge detection  
    # requires input image to be of mode = Greyscale (L) 
    image = image.convert("L") 
      
    # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES 
    image = image.filter(ImageFilter.FIND_EDGES) 
      
    # Saving the Image Under the name Edge_Sample.png 
    image.save(filename)



    # Opening the image (R prefixed to string 
    # in order to deal with '\' in paths) 
    image2 = Image.open(filename2) 
      
    # Converting the image to greyscale, as edge detection  
    # requires input image to be of mode = Greyscale (L) 
    image2 = image2.convert("L") 
      
    # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES 
    image2 = image2.filter(ImageFilter.FIND_EDGES) 
      
    # Saving the Image Under the name Edge_Sample.png 
    image2.save(filename2)


    #motor de clasificación
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    size = classify.input_size(interpreter)
    image = Image.open(img_input).convert('RGB').resize(size, Image.ANTIALIAS)
    classify.set_input(interpreter, image)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("")
    print('----INFERENCE TIME----')
    print('Note: The first inference on Edge TPU is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(5):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      classes = classify.get_output(interpreter, 1, 0)
      print('%.1fms' % (inference_time * 1000))

    print('-------RESULTS--------')
    for klass in classes:
      print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))

    #segunda imagen
    
    size = classify.input_size(interpreter)
    image2 = Image.open(img_input2).convert('RGB').resize(size, Image.ANTIALIAS)
    classify.set_input(interpreter, image2)
    print("")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("")
    print('----INFERENCE TIME 2----')
    print('Note: The first inference on Edge TPU is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(5):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      classes = classify.get_output(interpreter, 1, 0)
      print('%.1fms' % (inference_time * 1000))

    print('-------RESULTS2--------')
    for klass in classes:
      print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))



    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    # Window name in which image is displayed 
    image_final = cv2.imread(img_input2) 
    window_name = ''
   
    # Using cv2.imshow() method  
    # Displaying the image  
    cv2.imshow(window_name, image_final) 
        
    #closing all open windows  
  
  cv2.destroyAllWindows()   

if __name__ == '__main__':
  main()