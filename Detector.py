import cv2  
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
import requests # to get image from the web
import shutil # to save it locally
import json

def main():

  #configuraciones
  objetos=20
  limite=0.1
  limite_racimo=5
  model='model.tflite'
  labels='labels.txt'
  img_input='input.jpg'
  img_output='output.jpg'


  # Initialize engine.
  engine = DetectionEngine(model)
  labels = dataset_utils.read_label_file(labels)

  while cv2.waitKey(1) & 0xFF != ord('q'):


    ret,frame = cap.read() # return a single frame in variable `frame`
    cv2.imwrite(img_input,frame)   



    # Save image with bounding boxes.
    if img_output:

      img = img.resize((round(img.size[0]*3), round(img.size[1]*3)))
      img.save(img_output)


      

      # Open image.
      img = Image.open(img_input).convert('RGB')
      #Make the new image half the width and half the height of the original image
      img = img.resize((round(img.size[0]), round(img.size[1])))
  
      draw = ImageDraw.Draw(img)
    
      # Run inference.
      objs = engine.detect_with_image(img,
                                      threshold=limite,
                                      keep_aspect_ratio='store_true',
                                      relative_coord=False,
                                      top_k=objetos)

      # Print and draw detected objects.
      print('---------------- OBJETO 1 ----------------')
      for obj in objs:
        if labels:
          if(labels[obj.label_id] == "AMBAR"):
            AMBAR=AMBAR+1
          if(labels[obj.label_id] == "VERDE"):
            VERDE=VERDE+1
        
        box = obj.bounding_box.flatten().tolist()
        #print('box =', box)
        draw.rectangle(box, outline='yellow')
  
      if not objs:
        print('No objects detected.')


      #generando archivo de salida
      OUTPUT1_calibre='Detectando..'
      OUTPUT2_calibre='Detectando..'

      data = {'color':{'CAM1':OUTPUT1,'CAM2':OUTPUT2}, 'calibre':{'CAM1':OUTPUT1_calibre,'CAM2':OUTPUT2_calibre}}

      with open('/var/www/html/data.json', 'w') as outfile:
          json.dump(data, outfile)


      # Save image with bounding boxes.
      if img_output :
        img.save(img_output)

        # Reading an image in default mode 
        image = cv2.imread(img_output) 

        image_final = cv2.imread("final.jpg") 

        # Window name in which image is displayed 
        window_name = ''
            
        # Using cv2.imshow() method  
        # Displaying the image  
        cv2.imshow(window_name, image_final) 
            
        #closing all open windows  
  cv2.destroyAllWindows()      

if __name__ == '__main__':
  main()

