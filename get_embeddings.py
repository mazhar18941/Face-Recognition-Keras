import argparse
import pickle, os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import face_net

def get_embeddings(database_folder_path, facenet_detector):
    file_names = [f for f in os.listdir(database_folder_path) if f.split('.')[1] == 'jpg' or 'jpeg' or 'JPG' or 'JPEG']
    embed_dic = {}
    for file_name in file_names:
        img_path = database_folder_path + '/' + file_name
        img = load_img(img_path)
        img = img_to_array(img)
        person_name = file_name.split('.')[0]
        embed_dic[person_name] = face_net.extract_embeddings(img, facenet_detector)
    return embed_dic

def save_embeddings(embed_dic, path='embedding_dict'):
    with open(path, 'wb') as f:
        pickle.dump(embed_dic, f)

def load_embeddings(path='embedding_dict'):
    with open(path, 'rb') as f:
        embd = pickle.load(f)
    return embd


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--database_folder_path", required=True,
                    default="database_images/",
                    help="path to input image")
    ap.add_argument("-md", "--vgg_facenet_model", type=str,
                    default="models/vgg_facenet_model.h5",
                    help="path to trained vgg facenet model")

    args = vars(ap.parse_args())

    facenet_model_path = args['vgg_facenet_model']
    database_folder_path = args['database_folder_path']
    embedding_path = 'database_embedding_dict'

    facenet_detector = face_net.load_model(facenet_model_path)

    embed_dic=get_embeddings(database_folder_path, facenet_detector)
    save_embeddings(embed_dic,path=embedding_path)
    #embed = load_embeddings(path=embedding_path)

    print('Embeddings are saved to '+os.getcwd()+'/'+embedding_path)
