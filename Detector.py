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
  limite=0.4
  model='model.tflite'
  labels='labels.txt'
  img_input='input.jpg'
  img_output='output.jpg'

  img_input2='input2.jpg'
  img_output2='output2.jpg'

  # Set up the image URL and filename
  image_url = "http://192.168.1.10:8080/?action=snapshot"
  image_url2 = "http://192.168.1.20:8080/?action=snapshot"

  #final configuraciones

  filename =  img_input
  filename2 =  img_input2

  # Initialize engine.
  engine = DetectionEngine(model)
  labels = dataset_utils.read_label_file(labels)

  while cv2.waitKey(1) & 0xFF != ord('q'):

    AMBAR=0
    AMBAR_SUAVE=0
    VERDE=0
    CREMA=0

    AMBAR2=0
    AMBAR_SUAVE2=0
    VERDE2=0
    CREMA2=0
    
    print()
    print('____________________________________')
    print()
    
    reintento=1
    # Open the url image, set stream to True, this will return the stream content.
    while reintento:
      try: 
        r = requests.get(image_url, stream = True)
        r2 = requests.get(image_url2, stream = True)
        reintento=0
      except:
        print("Error al descagar imagenes.. reintento "+str(reintento))
        reintento=reintento+1
         

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
    
    #recortes de imagenes segun camara

    im = Image.open(filename)
    im = im.crop((50, 100, 500, 300))
    im.save(filename)

    im2 = Image.open(filename2)
    im2 = im2.crop((150, 100, 500, 300))
    im2.save(filename2)
   

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
        #print(labels[obj.label_id])
        #print('score =', obj.score)
        if(labels[obj.label_id] == "AMBAR"):
           AMBAR=AMBAR+1
        if(labels[obj.label_id] == "AMBAR_SUAVE"):
           AMBAR_SUAVE=AMBAR_SUAVE+1
        if(labels[obj.label_id] == "VERDE"):
           VERDE=VERDE+1
        if(labels[obj.label_id] == "CREMA"):
           CREMA=CREMA+1
      
    
      box = obj.bounding_box.flatten().tolist()
      #print('box =', box)
      draw.rectangle(box, outline='yellow')
    TOTAL1=AMBAR+AMBAR_SUAVE+VERDE+CREMA

    print('AMBAR: '+str(AMBAR))
    print('AMBAR_SUAVE: '+str(AMBAR_SUAVE))
    print('VERDE: '+str(VERDE))
    print('CREMA: '+str(CREMA))
    print()
    print('TOTAL: '+str(TOTAL1))

    if not objs:
      print('No objects detected.')

    # Open image.
    img2 = Image.open(img_input2).convert('RGB')
    #Make the new image half the width and half the height of the original image
    img2 = img2.resize((round(img2.size[0]), round(img2.size[1])))
 
    draw2 = ImageDraw.Draw(img2)
   
    # Run inference.
    objs2 = engine.detect_with_image(img2,
                                    threshold=limite,
                                    keep_aspect_ratio='store_true',
                                    relative_coord=False,
                                    top_k=objetos)

    # Print and draw detected objects.
    print('---------------- OBJETO 2 ----------------')
    for obj2 in objs2:
      if labels:
        if(labels[obj2.label_id] == "AMBAR"):
           AMBAR2=AMBAR2+1
        if(labels[obj2.label_id] == "AMBAR_SUAVE"):
           AMBAR_SUAVE2=AMBAR_SUAVE2+1
        if(labels[obj2.label_id] == "VERDE"):
           VERDE2=VERDE2+1
        if(labels[obj2.label_id] == "CREMA"):
           CREMA2=CREMA2+1
      
      box2 = obj2.bounding_box.flatten().tolist()
      #print('box =', box)
      draw2.rectangle(box2, outline='yellow')

    TOTAL2=AMBAR2+AMBAR_SUAVE2+VERDE2+CREMA2
    print('AMBAR: '+str(AMBAR2))
    print('AMBAR_SUAVE: '+str(AMBAR_SUAVE2))
    print('VERDE: '+str(VERDE2))
    print('CREMA: '+str(CREMA2))
    print()
    print('TOTAL: '+str(TOTAL2))


    if not objs2:
      print('No objects detected.')   

    #clasificando racimos
    print()
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print()

    if (CREMA == 0 and AMBAR== 0 and AMBAR_SUAVE == 0):
      print("RACIMO 1 ES ----->    VERDE")
      OUTPUT1='VERDE'
    if (CREMA >= 1 and AMBAR ==0  and AMBAR_SUAVE == 0):
      print("RACIMO 1 ES ----->    CREMA")
      OUTPUT1='CREMA'
    if (AMBAR_SUAVE >= 1  and AMBAR == 0):
      print("RACIMO 1 ES ----->    AMBAR_SUAVE")
      OUTPUT1='AMBAR SUAVE'
    if (AMBAR >= 1):
      print("RACIMO 1 ES ----->    AMBAR")
      OUTPUT1='AMBAR'

    if (CREMA2 == 0 and AMBAR2== 0 and AMBAR_SUAVE2 == 0):
      print("RACIMO 2 ES ----->    VERDE")
      OUTPUT2='VERDE'
    if (CREMA2 >= 1 and AMBAR2 ==0  and AMBAR_SUAVE2 == 0):
      print("RACIMO 2 ES ----->    CREMA")
      OUTPUT2='CREMA'
    if (AMBAR_SUAVE2 >= 1  and AMBAR2 == 0):
      print("RACIMO 2 ES ----->    AMBAR_SUAVE")
      OUTPUT2='AMBAR SUAVE'
    if (AMBAR2 >= 1):
      print("RACIMO 2 ES ----->    AMBAR")
      OUTPUT2='AMBAR'
   
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    if TOTAL1 < 5:
        OUTPUT1='Detectando..'
    if TOTAL2 < 5:
        OUTPUT2='Detectando..

    #generando archivo de salida
    OUTPUT1_calibre='Detectando..'
    OUTPUT2_calibre='Detectando..'
    
    data = {'color':{'CAM1':OUTPUT1,'CAM2':OUTPUT2}, 'calibre':{'CAM1':OUTPUT1_calibre,'CAM2':OUTPUT2_calibre}}

    with open('/var/www/html/data.json', 'w') as outfile:
        json.dump(data, outfile)


    # Save image with bounding boxes.
    if img_output or img_output2:
      img.save(img_output)

      # Reading an image in default mode 
      image = cv2.imread(img_output) 

      img2.save(img_output2)

      # Reading an image in default mode 
      image2 = cv2.imread(img_output2)
  
      #mostrar imagenes en una ventana!
      final = Image.new("RGB",(1000,250),"black")
      imagen1 = Image.open(img_output)
      imagen2 = Image.open(img_output2)
      final.paste(imagen1, (0,0))
      final.paste(imagen2, (500,0))
      final.save("final.jpg")

      image_final = cv2.imread("final.jpg") 

      # Window name in which image is displayed 
      window_name = ''
           
      # Using cv2.imshow() method  
      # Displaying the image  
      #cv2.imshow(window_name, image_final) 
          
      #closing all open windows  
  cv2.destroyAllWindows()      

if __name__ == '__main__':
  main()

