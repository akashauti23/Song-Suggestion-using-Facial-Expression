from config import *
#from Dataloader import *
from model import *
#from train import *
import playsound
from pygame import mixer  # Load the popular external library
import time

model_fer = load_model(weights)
model_fer.load_weights(weights)

images = []
for filename in os.listdir(inferenceDir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(inferenceDir, filename)
        image = Image.open(image_path).convert("RGB")
        # preprocess the image
        image = image.resize((224, 224)) # example resizing to 224x224
        image = np.array(image)
        images.append(image)
images = np.array(images)
print(f'list of images :: {images.shape}')

# Make predictions on the images using the model
A=model_fer.predict(images)


FER=[]
for i in range(1):
    predicted_class = np.argmax(A[i])
    fer= ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    z=fer[predicted_class]
    FER.append(z)
    
#print(FER)

# Display the predicted age and gender for each image
for i in range(len(images)):
    plt.imshow(images[i])
    plt.title(f"Predicted Emotion: {FER[i]} ")
    plt.show()
    
    
for i in range(len(images)):
   
    if FER[i] == 'happy' or FER[i] == 'neutral' or FER[i]=='surprise':
        os.chdir(HappySongDir)

        files=os.listdir()
        song = random.choice(files)        
        
        print('\n\nplaying happy songs\n\n')
        print(f'list of songs\n{files}\n\n')
        
        
        #os.startfile(song)
        
        #playsound.playsound(song, True)
        

        mixer.init()
        mixer.music.load(song)
        mixer.music.play()

        print(f'currently playing :- {song}')
        time.sleep(10)
        
        
    elif FER[i] == 'sad' or FER[i] == 'angry' or FER[i] == 'fear' or FER[i] == 'disgust':
        
        os.chdir(SadSongDir)

        
        files=os.listdir()
        song = random.choice(files)        
        
        print('\n\nplaying sad songs\n\n')
        print(f'list of songs\n{files}\n\n')
        
	#os.startfile(song)
        #playsound.playsound(song, True)
        mixer.init()
        mixer.music.load(song)
        mixer.music.play()
        time.sleep(10)
        
        print(f'currently playing :- {song}')
        
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

