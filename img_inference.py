import argparse
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import get_embeddings
import face_net, face_detection

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	default="images/image.jpg",
	help="path to input image")

ap.add_argument("-e", "--database_embedding_path", required=False,
	default="database_embedding_dict",
	help="path to database embedding dictionary")

ap.add_argument("-fm", "--face_model", type=str,
	default="models/deploy.prototxt.txt",
	help="path to face detector model")

ap.add_argument("-fw", "--face_weights", type=str,
	default="models/res10_300x300_ssd_iter_140000.caffemodel",
	help="path to face detector model weights")

ap.add_argument("-md", "--vgg_facenet_model", type=str,
	default="models/vgg_facenet_model.h5",
	help="path to trained vgg facenet model")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-d", "--max_distance", type=float, default=0.35,
	help="maximum similarity for face recognition")

args = vars(ap.parse_args())


caffeModel = args['face_model']
prototextPath = args['face_weights']
facenet_model_path = args['vgg_facenet_model']
img_tobe_verfied_path = args['image']
database_embedding_path = args['database_embedding_path']
threshold = args['confidence']
max_distance = args['max_distance']


face_detector = face_detection.load_model(prototextPath, caffeModel)
facenet_detector = face_net.load_model(facenet_model_path)

img_tobe_verfied = load_img(img_tobe_verfied_path)
img_tobe_verfied = img_to_array(img_tobe_verfied)

detections = face_detection.get_detection(face_detector, img_tobe_verfied)
bb_coord_pixel = face_detection.get_face_pixels(img_tobe_verfied, detections, threshold=threshold)

new_size = facenet_detector.layers[0].input_shape[0][1:3]

def verify_from_database(img_tobe_verfied, facenet_detector, database_embedding_path, max_distance=0.45):
    img_embedding = face_net.extract_embeddings(img_tobe_verfied, facenet_detector)
    database_embeddings_dict = get_embeddings.load_embeddings(path=database_embedding_path)
    ord_list = face_net.compare_embeddings(img_embedding, database_embeddings_dict)
    if ord_list[0][1] < max_distance:
        label = ord_list[0][0].split('.')[0]+' '+str(round(((1-ord_list[0][1])*100),1))

    else:
        label = 'unknown'

    return label

def get_bb_label(bb_coord_pixel):
    bb_dic = {'bb_coord': bb_coord_pixel[0], 'bb_pixel': bb_coord_pixel[1],
              'bb_label': []}

    for bb_pixel in bb_dic['bb_pixel']:
        # resized_img = image.resize_image(bb_pixel, new_size=new_size)
        label = verify_from_database(img_tobe_verfied=bb_pixel, facenet_detector=facenet_detector,
                                     database_embedding_path=database_embedding_path,
                                     max_distance=max_distance)
        bb_dic['bb_label'].append(label)

    return bb_dic

coord_pixels_label = get_bb_label(bb_coord_pixel)

img = cv2.imread(img_tobe_verfied_path)
if coord_pixels_label:
    for coord, label in zip(coord_pixels_label['bb_coord'],
                            coord_pixels_label['bb_label']):
        (startX, startY, endX, endY) = coord
        text = label

        r_clr = (0, 255, 0)
        t_clr = (0, 0, 255)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY),
                      r_clr, 2)
        cv2.putText(img, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_clr, 1)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Face Recognition', img)
    print('***PRESS ANY KEY TO QUIT***')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #filename = 'savedImage1.jpg'
    #cv2.imwrite(filename, img)

else:
    print('Face was not detected in image')