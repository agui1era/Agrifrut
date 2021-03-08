import cv2  
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
import requests # to get image from the web
import shutil # to save it locally

NEGRA=0
ROSADA=0

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

    rosada=0
    negra=0
    # Open the url image, set stream to True, this will return the stream content.
  
    ret,frame = cap.read() # return a single frame in variable `frame`
    cv2.imwrite(img_input,frame)   
   
    # Open image.
    img = Image.open(img_input).convert('RGB')
    #Make the new image half the width and half the height of the original image
    img = img.resize((round(img.size[0]*0.5), round(img.size[1])))
 
    draw = ImageDraw.Draw(img)
   
    # Run inference.
    objs = engine.detect_with_image(img,
                                    threshold=0.5,
                                    keep_aspect_ratio='store_true',
                                    relative_coord=False,
                                    top_k=10)

    # Print and draw detected objects.
    for obj in objs:
      if labels:
        print(labels[obj.label_id])
        if(labels[obj.label_id] == "negra"):
            negra=negra+1
        if(labels[obj.label_id] == "rosada"):
            rosada=rosada+1
        
      box = obj.bounding_box.flatten().tolist()
      draw.rectangle(box, outline='yellow')

    if not objs:
      print('No objects detected.')
    print('__________________________________')
    print('')
    print('TOTAL negra: '+str(negra))
    print('TOTAL rosada:'+str(rosada))
    print('__________________________________')
    print('')

    # Save image with bounding boxes.
    if img_output:
      img.save(img_output)


      image = cv2.imread(img_output) 
      cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
      #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
      #cv2.imshow("window",image)

      #closing all open windows  
  cv2.destroyAllWindows()   
  cap.release()   

if __name__ == '__main__':
  main()
  