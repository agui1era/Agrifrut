
import cv2  
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
import requests # to get image from the web
import shutil # to save it locally


def main():
  model='model.tflite'
  labels='labels.txt'
  img_input='input.jpg'
  img_output='output.jpg'

  cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
  filename =  img_input

  # Initialize engine.
  engine = DetectionEngine(model)
  labels = dataset_utils.read_label_file(labels) 


  while cv2.waitKey(1) & 0xFF != ord('q'):
    # Open the url image, set stream to True, this will return the stream content.
  
    ret,frame = cap.read() # return a single frame in variable `frame`
    cv2.imwrite(img_input,frame)   
   
    # Open image.
    img = Image.open(img_input).convert('RGB')
    #Make the new image half the width and half the height of the original image
    img = img.resize((round(img.size[0]*0.5), round(img.size[1]*0.5)))
 
    draw = ImageDraw.Draw(img)
   
    # Run inference.
    objs = engine.detect_with_image(img,
                                    threshold=0.5,
                                    keep_aspect_ratio='store_true',
                                    relative_coord=False,
                                    top_k=10)

    # Print and draw detected objects.
    for obj in objs:
      print('-----------------------------------------')
      if labels:
        print(labels[obj.label_id])
      print('score =', obj.score)
      box = obj.bounding_box.flatten().tolist()
      print('box =', box)
      draw.rectangle(box, outline='yellow')

    if not objs:
      print('No objects detected.')

    # Save image with bounding boxes.
    if img_output:
      img.save(img_output)

      image = Image.open(img_output)
      new_image = image.resize((960, 720))
      new_image.save(img_output)

      #concatenando imagnes
      im1 = cv2.imread(img_output) 
      im2 = cv2.imread('ffffff.png')
      im3 = cv2.imread('ffffff.png')
      

      font = cv2.FONT_HERSHEY_SIMPLEX 
  
      # org 
      org = (100, 100) 
        
      # fontScale 
      fontScale = 2
        
      # Blue color in BGR 
      color = (255, 0, 0) 
        
      # Line thickness of 2 px 
      thickness = 2
        
      # Using cv2.putText() method 
      cv2.putText(im2, 'AMBAR', org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 

      org = (100, 200) 

         # Using cv2.putText() method 
      cv2.putText(im2, 'XL', org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
      org = (100, 100) 
        
      cv2.putText(im3, 'VERDE', org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
      org = (100, 200) 

      cv2.putText(im3, 'S', org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
                 
  
      im_h = cv2.hconcat([im2, im1])

      im_h = cv2.hconcat([im_h, im3])
      cv2.imwrite(img_output, im_h)

      image = cv2.imread(img_output) 

     
      cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
      cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
      cv2.imshow("window",image)

      

          
      #closing all open windows  
  cv2.destroyAllWindows()   
  cap.release()   

if __name__ == '__main__':
  main()
  

