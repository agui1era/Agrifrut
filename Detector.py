
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

  img_input2='input2.jpg'
  img_output2='output2.jpg'

  # Set up the image URL and filename
  image_url = "http://192.168.0.6:8080/?action=snapshot"
  image_url2 = "http://192.168.0.3:8080/?action=snapshot"

  filename =  img_input
  filename2 =  img_input2

  # Initialize engine.
  engine = DetectionEngine(model)
  labels = dataset_utils.read_label_file(labels) 

  while cv2.waitKey(1) & 0xFF != ord('q'):
    
    print()
    print('____________________________________')
    print()

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)
    r2 = requests.get(image_url2, stream = True)

    # Check if the image was retrieved successfully
    if (r.status_code == 200) and (r2.status_code == 200) :
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        r2.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded1 : ',filename)

        # Open a local file with wb ( write binary ) permission.
        with open(filename2,'wb') as f2:
            shutil.copyfileobj(r2.raw, f2)
            
        print('Image sucessfully Downloaded2 : ',filename2)


    else:
        print('Images Couldn\'t be retreived')
    
    im = Image.open(filename)
    im = im.crop((150, 100, 500, 300))
    im.save(filename)

    im2 = Image.open(filename2)
    im2 = im2.crop((150, 100, 500, 300))
    im2.save(filename2)
   

    # Open image.
    img = Image.open(img_input).convert('RGB')
    #Make the new image half the width and half the height of the original image
    img = img.resize((round(img.size[0])*2, round(img.size[1])*2))
 
    draw = ImageDraw.Draw(img)
   
    # Run inference.
    objs = engine.detect_with_image(img,
                                    threshold=0.5,
                                    keep_aspect_ratio='store_true',
                                    relative_coord=False,
                                    top_k=10)

    # Print and draw detected objects.
    print('---------------- OBJETO 1 ----------------')
    for obj in objs:
      if labels:
        print(labels[obj.label_id])
      print('score =', obj.score)
      box = obj.bounding_box.flatten().tolist()
      #print('box =', box)
      draw.rectangle(box, outline='yellow')

    if not objs:
      print('No objects detected.')


   
    # Open image.
    img2 = Image.open(img_input2).convert('RGB')
    #Make the new image half the width and half the height of the original image
    img2 = img2.resize((round(img2.size[0])*2, round(img2.size[1])*2))
 
    draw2 = ImageDraw.Draw(img2)
   
    # Run inference.
    objs2 = engine.detect_with_image(img,
                                    threshold=0.5,
                                    keep_aspect_ratio='store_true',
                                    relative_coord=False,
                                    top_k=10)

    # Print and draw detected objects.
    print('---------------- OBJETO 2 ----------------')
    for obj2 in objs2:
      if labels:
        print(labels[obj2.label_id])
      print('score =', obj2.score)
      box2 = obj2.bounding_box.flatten().tolist()
      #print('box =', box)
      draw.rectangle(box2, outline='yellow')

    if not objs2:
      print('No objects detected.')   



    #clasificando racimos


  



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

