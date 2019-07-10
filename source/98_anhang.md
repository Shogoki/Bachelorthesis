# Anhang {.unnumbered}
\chaptermark{Anhang}
\renewcommand{\thesection}{\Alph{section}}

## Quellcode 

### Vorverarbeitung selbstaufgezeichnete Daten \label{anhang_prep_selfrecorded}

```python
# code to prepare a single image from selfrecorded dataset
def prep_single_img_selfrecorded(img,
		viola_jones_model="viola_jones.xml",
		image_shape=(48, 48), grey=True):

    face_detection = cv2.CascadeClassifier(viola_jones_model)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face detection 
    faces = face_detection.detectMultiScale(grey_img,
		scaleFactor=1.1, minNeighbors=5, minSize=(20,20),
		flags=cv2.CASCADE_SCALE_IMAGE)
    # sorting out images with failed detection
    target_face = None
    if len(faces) == 0:
        print ("ERROR: there was no face detected on image,",
			"discarding it")
        return None
    elif len(faces) > 1:
        print ("WARNING: there was more than one face", 
			"detected on image, using the biggest one")
		# finding face with biggest size
        for face in faces:
            size = face[2] * face[3]
            if target_face is None or (
                        target_face[2] * target_face[3]
						< size):
                target_face = face
    else:
        target_face = faces[0]
    # crop image to face & turn to greyscale if specified
    (fX, fY, fW, fH) = target_face
    if grey:
        img = grey_img[ fY: fY + fH, fX: fX +fW].copy()
    else:
        img = img[ fY: fY + fH, fX: fX +fW].copy()
    # rescale it to destination size
    img = cv2.resize(img.astype('uint8'),image_shape)
	# return image as float matrix
    return img.astype('float32')
```

```python
## prepare complete dataset
def prep_data_selfrecorded(selfrecorded_data,
	viola_jones_model="viola_jones.xml",
	image_shape=(48, 48), grey=True):
    
	#get list of all images in dataset
    img_input = selfrecorded_data['img'].tolist()
    # declare lists for later use
	imgs = []
    no_face = []
	# loop through all images by index i
    for i in range(len(img_input)):
		# prepare current image
        img = prep_single_img_selfrecorded(img_input[i],
			viola_jones_model, image_shape, grey)
		# if img is empty there was no face detected
        if img is None:
			# save current index in no_face list
            no_face.append(i)
        else:
			# otherwise append img to imgs list
            imgs.append(img)
	# turn the list into numpy array
    imgs = np.asarray(imgs)
    if grey:
	# if we converted to greyscale we need to add dimension
        imgs = np.expand_dims(imgs, -1)
    # dropping all failed detections from DataFrame
    selfrecorded_data = selfrecorded_data.drop(no_face)
    # returning emotions OneHot Encoded. 
    emo = pd.get_dummies(selfrecorded_data['emotion'])
    # we need to make sure, to use always the same cols
    cols = ['anger', 'contempt', 'disgust', 'fear',
		'happiness', 'neutral', 'sadness', 'surprise' ]
    for col in cols:
        if col not in emo.columns.tolist():
            # if col not present create and fill with zeros
            emo[col] = 0
    return imgs, emo[cols].as_matrix()
```
\newpage

### Vorverarbeitung der FER+ Daten \label{anhang_prep_ferplus}

```python
def prep_data_ferplus(ferplus_data,
    image_shape=(48, 48), grey=True):
    
    # removing NF (no face) and unknown columns
    cleaned = ferplus_data[ferplus_data['emotion'] != "NF"]
    cleaned = cleaned[cleaned['emotion'] != "unknown"]
    # getting oneHot encoded classes using get_dummies
    emotions = pd.get_dummies(cleaned['emotion']).as_matrix()
    # retrieving images from pixel list
    img_pixels = cleaned['img_pixels'].tolist()
    # original image sizes from dataset
    img_width, img_height = 48, 48
    imgs = []
    # extracting images from space delimited pixel values
    for pixel_row in img_pixels:
        img = [int(pixel) for pixel in pixel_row.split(' ')]
        # having them in a one-dim list now,
        # we have to make an np array with our img_shape
        img = np.asarray(img).reshape(img_width, img_height)
        img = cv2.resize(img.astype('uint8'),image_shape)
        # the original image is greyscale
        if not grey:
        # so if we dont want that we have to convert
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # and add them to our list
        imgs.append(img.astype('float32'))
    # finally we make an numpy array out of it
    imgs = np.asarray(imgs)
    if grey:
    #and if it is greyscale we add one dimension
        imgs = np.expand_dims(imgs, -1)
    # returning our prepared data
    return imgs, emotions
```


### Emotions-Zuordnung der Einzelbilder \label{anhang_pred_to_emotion}

```python
def pred_to_text(pred, cols = ['anger', 'contempt',
         'disgust', 'fear', 'happiness', 'neutral', 
         'sadness', 'surprise' ]):
    
    # getting the emotion label for the highest predicition
    highest = 0.0
    emotion = ""
    # loop through predictions
    for i in range(len(pred['predictions'][0])):
        if(highest < pred['predictions'][0][i]):
            # store emotion label for highest prediction
            highest = pred['predictions'][0][i]
            emotion = cols[i]
    return emotion
```

\newpage

```python
def predict_emotion(image):
    # prepare image data
    img = prep_single_img_selfrecorded(image,
        image_shape=(INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT),
            grey=INPUT_IMG_GREY)
    # if no image returned no face was detected
    if img is None:
        return "no_face"
    # normalize the image
    img = normalize_input(img)
    #create payload for classifier
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }
    # sending post request to TensorFlow Serving server
    r = requests.post(TF_CLASSIFIER_URL, json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    return pred_to_text(pred)
```

\newpage

## Sonstiges

### Docker-Compose Datei \label{anhang_compose}

```yaml
version: '3.0'
services:
 api:
  image: skraus/emopred-api
  build: ./api
  environment:
   TF_CLASSIFIER_URL: http://tfs:8501/v1/models/model:predict
  ports:
   - 8080:8080
 tfs:
  image: skraus/emopred-classifier
  build: ./classifier
  environment:
   MODEL_NAME: model
```
\newpage

### Rückgabewerte des Webservice \label{anhang_websvc_json}

```JSON
{
  "emotions": [
    { "emotion": "happiness", "time": 0 },
    { "emotion": "happiness", "time": 1 },
	{ "emotion": "no_face",   "time": 2 },
	{ "emotion": "no_face",   "time": 3 },
	{ "emotion": "no_face",   "time": 4 },
	{ "emotion": "happiness", "time": 5 },
    { "emotion": "happiness", "time": 6 },
    { "emotion": "happiness", "time": 7 },
    { "emotion": "happiness", "time": 8 },
    { "emotion": "happiness", "time": 9 },
    { "emotion": "happiness", "time": 10},
    { "emotion": "happiness", "time": 11},
    { "emotion": "happiness", "time": 12},
    { "emotion": "happiness", "time": 13}
  ],
  "videoname": "happiness_55zc2lbb1o7y5s3y37.webm"
}
```


\newpage