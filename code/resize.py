import os, cv2
from PIL import Image
#ONLY RUN ONCE
#RESIZE IMAGE AND SPLIT THE DATASET INTO TRAIN AND TEST SETS (RATIO: 70/30)
def prepare_dataset():
  
  #dir path of dataset
  df = './raw-img/' #folder to store raw images
  dataset_dir = './dataset/'
  
  #size of image
  size = (128,128)  #because we use CNN with pooling layers (powers of 2) and our computer configuration => We choose that value of size
  
  #create dataset folder if not exists
  if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
  
  #create train folder if not exists
  if not os.path.exists(dataset_dir+'train'):
    os.makedirs(dataset_dir+'train')
  
  #create test folder if not exists
  if not os.path.exists(dataset_dir+'test'):
    os.makedirs(dataset_dir+'test')
  
  for i in os.listdir(df):
    print('-->' + i)
    
    #use to run 1 class only
    #number of files in 1 class
    num_file = (len([name for name in os.listdir(df+i) if os.path.isfile(df+i+'/'+name)]))
    
    #number of test files in 1 class
    num_test = int(round(num_file * 0.3))
    
    #number of train files in 1 class
    num_train = num_file - num_test
    
    print("total, test, train:", num_file,num_test,num_train)
    
    #create class folder if not exists
    if not os.path.exists(dataset_dir+'train/'+i):
      os.makedirs(dataset_dir+'train/'+i)
    if not os.path.exists(dataset_dir+'test/'+i):
      os.makedirs(dataset_dir+'test/'+i)
      
    #idx variable
    cnt = 0
    for j in os.listdir(df+i):
      cnt += 1
      
      img = Image.open(df + i + '/' + j)
      
      #resize image and save in correct folder
      img = img.resize(size, Image.ANTIALIAS)
      
      if cnt <= num_train:
        img.save(dataset_dir + 'train/' + i + '/' + j,'JPEG')
      else:
        img.save(dataset_dir + 'test/' + i + '/' + j,'JPEG')
        
      if (cnt%100 == 0):
        print(cnt)
        
    print('finish '+ i + ":", cnt)
