# Example of shape predictors training.

import cv2
import dlib
import os
import re

# regex:
REG_PART = re.compile("part name='[0-9]+'")
REG_NUM = re.compile("[0-9]+")

# landmarks subsets (relative to 68-landmarks):
EYE_EYEBROWS = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47]
NOSE_MOUTH = [27, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52,
              53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
FACE_CONTOUR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ALL_LANDMARKS = range(0, 68)

# download the iBug 300W dataset
# wget.download("http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz", "ibug_300w.tar.gz")

# then unpack 'ibug_300w.tar.gz' archive and put the script in the same folder.

# dataset path
ibug_dir = "ibug_300W_large_face_landmark_dataset"

# annotations
train_labels ="labels_ibug_300W_train.xml"
test_labels = "labels_ibug_300W_test.xml"

def slice_xml(in_path, out_path, parts):
    '''creates a new xml file stored at [out_path] with the desired landmark-points.
    The input xml [in_path] must be structured like the ibug annotation xml.'''
    file = open(in_path, "r")
    out = open(out_path, "w")
    pointSet = set(parts)

    for line in file.readlines():
        finds = re.findall(REG_PART, line)

        # find the part section
        if len(finds) <= 0:
            out.write(line)
        else:
            # we are inside the part section 
            # so we can find the part name and the landmark x, y coordinates
            name, x, y = re.findall(REG_NUM, line)

            # if is one of the point i'm looking for, write in the output file
            if int(name) in pointSet:
                out.write(f"      <part name='{name}' x='{x}' y='{y}'/>\n")

    out.close()

def train_model(name, xml):
  '''requires: the model name, and the path to the xml annotations.
  It trains and saves a new model according to the specified 
  training options and given annotations'''
  # get the training options
  options = dlib.shape_predictor_training_options()
  options.tree_depth = 3
  options.nu = 0.1
  options.cascade_depth = 10
  options.feature_pool_size = 150
  options.num_test_splits = 350 
  options.oversampling_amount = 5
  options.oversampling_translation_jitter = 0

  options.be_verbose = True  # tells what is happening during the training
  options.num_threads = 1    # number of the threads used to train the model
  
  # finally, train the model
  dlib.train_shape_predictor(xml, name, options)

  
def measure_model_error(model, xml_annotations):
    '''requires: the model and xml path.
    It measures the error of the model on the given
    xml file of annotations.'''
    error = dlib.test_shape_predictor(xml_annotations, model)
    print("Error of the model: {} is {}".format(model, error))

# -----------------------------------------------------------------------------
# -- Model Generation
# -----------------------------------------------------------------------------

# add or remove models here.
models = [
     # pair: model name, parts
    ("eye_eyebrows_22", EYE_EYEBROWS),
    ("nose_mouth_30", NOSE_MOUTH),
    ("face_contour_17", FACE_CONTOUR),
    ("face_landmarks_68", ALL_LANDMARKS)
]

for model_name, parts in models:
  print(f"processing model: {model_name}")
  
  train_xml = f"{model_name}_train.xml"
  test_xml = f"{model_name}_test.xml"
  dat = f"{model_name}.dat"
  slice_xml(train_labels, train_xml, parts)
  slice_xml(test_labels, test_xml, parts)
  
  # training
  train_model(dat, train_xml)
  
  # compute traning and test error
  measure_model_error(dat, train_xml)
  measure_model_error(dat, test_xml)

# -----------------------------------------------------------------------------

def test(image_path, model_path):
    '''Test the given model by showing the detected landmarks.
        - image_path: the path of an image. Should contain a face.
        - model_path: the path of a shape predictor model.
    '''
    image = cv2.imread(image_path)
    face_detector = dlib.get_frontal_face_detector()
    dets = face_detector(image, 1)
    predictor = dlib.shape_predictor(model_path)

    for d in dets:
      cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
      shape = predictor(image, d)
      for i in range(shape.num_parts):
        p = shape.part(i)
        cv2.circle(image, (p.x, p.y), 2, 255, 1)
        cv2.putText(image, str(i), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))

    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# uncomment to test
# test("test_image.jpg", "shape_predictor.dat")

