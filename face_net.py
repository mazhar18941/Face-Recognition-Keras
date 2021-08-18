import tensorflow as tf
import image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, smart_resize
from tensorflow.image import resize
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image


def load_model(path):
    return tf.keras.models.load_model(path)
    


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))



def preprocess_image(image):
    img = smart_resize(image, size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    #print(img.shape)
    return img



def verifyFace(img1, img2, vgg_face_model):
    img1_representation = vgg_face_model.predict(
        preprocess_image(img1))[0, :]
    img2_representation = vgg_face_model.predict(
        preprocess_image(img2))[0, :]

    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)

    #print("Cosine similarity: ", cosine_similarity)

    ''' if (cosine_similarity < epsilon):
        print("verified... they are same person")
        return True
    else:
        print("unverified! they are not same person!")
        return False '''
    return cosine_similarity

def extract_embeddings(img, vgg_face_model):
    img_representation = vgg_face_model.predict(
        preprocess_image(img))[0, :]
    return img_representation

def compare_embeddings(embed1, database_embeddings_dict):
    sim_dic = {}
    for name in database_embeddings_dict.keys():
        embed = database_embeddings_dict[name]
        sim_dic[name] = findCosineSimilarity(embed1, embed)
    ord_list = sorted(sim_dic.items(), key=lambda x: x[1])
    return ord_list

