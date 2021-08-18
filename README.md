# Face-Recognition-Keras
This repo contains code of face recognition system for images and videos.Bounding box and name of a person is drawn around his face if face image has similarity with the images in the database(images).Two models are used to recognize faces. Caffee face detection model detects faces in the image or frame and then these faces(pixels) are passed to keras facenet model which extracts 2622-dimensional embeddings for each face. These embeddings are compared to the pre-computed embeddings from database.If cosine similarity measure between face embedding and one of pre-computed face embeddings is less than a specific threshold, face is labelled as the label of that pre-computed face embedding. If similarity measure exceeds threshold, face is labelled as 'unknown'.

For example if a face embedding is compared to the pre-computed face embeddings of persons ranging from person_A to person_Z and cosine similarity with person_H is less than threshold(say 0.4) and least, face will be labelled as person_H.

## Installation and Working
1. weights of vgg_facenet model can be downloaded from [here](https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing).After weights file is downloaded, place it in **weights folder**.

2. clone this repo:

  `git clone https://github.com/mazhar18941/Face-Recognition-Keras.git`

3. install required packages:

  `pip install -r requirements.txt`

4. build and save keras vgg face model:

  `python build_model.py -w weights/vgg_face_weights.h5`

5. Now, put images of your friends or family members in **database_images folder**. Be sure that

- one person should have only one image in the folder

- only well-cropped faces should be included

- name of image should be the name of person in the image.For example if it's an image of **Alex**, image name should be **Alex.jpg** or **Alex.jpeg**.

6. Extract embeddings of database images and saving them in a dictionary:

   `python get_embeddings.py -p database_images/`
  
7. run inference on image:

   `python img_inference.py -i image_to_be_recognized.jpg`
   
8. run inference on webcam video:

   `python video_inference.py`
