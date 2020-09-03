
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

    ## Set up the image URL and filename
  image_url = "http://192.168.0.100:8080/?action=snapshot"
  filename =  img_input

  # Initialize engine.
  engine = DetectionEngine(model)
  labels = dataset_utils.read_label_file(labels) 

  while cv2.waitKey(1) & 0xFF != ord('q'):
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',filename)
    else:
        print('Image Couldn\'t be retreived')

   

    # Open image.
    img = Image.open(img_input).convert('RGB')
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

      # Reading an image in default mode 
      image = cv2.imread(img_output) 
      
      # Window name in which image is displayed 
      window_name = ''

      
      # Using cv2.imshow() method  
      # Displaying the image  
      cv2.imshow(window_name, image) 
          
      #closing all open windows  
  cv2.destroyAllWindows()      

if __name__ == '__main__':
  main()

