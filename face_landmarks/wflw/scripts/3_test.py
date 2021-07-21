import os
import cv2
import dlib
import matplotlib.pyplot as plt


def adjust_bounding_box(bbox, image, f=0.15):
    '''Enlarges the given bouding-box by `f`, interpreted as a percentage'''
    h, w = image.shape[:2]

    # determine spacing
    dw = f * (bbox.right() - bbox.left())
    dh = f * (bbox.bottom() - bbox.top())

    # compute new coordinates
    left = max(0, int(bbox.left() - dw))
    top = max(0, int(bbox.top() - dh))
    right = min(w, int(bbox.right() + dw))
    bottom = min(h, int(bbox.bottom() + dh))

    return dlib.rectangle(left, top, right, bottom)


def test_model(model_path: str, image_path: str, adjust=0.0, save=False):
    assert adjust >= 0.0

    # load shape predictor and get default face detector
    shape_predictor = dlib.shape_predictor(model_path)

    # NOTE: the accuracy of landmark detection is greatly affected by the 
    # size of the bouding-box of the detected faces. So, you may want to try
    # different face detectors, OR to resize it with `adjust_bounding_box()`
    face_detector = dlib.get_frontal_face_detector()

    # load image and detect faces
    image = cv2.imread(image_path)
    faces = face_detector(image, 1)

    print('detected faces:', len(faces))
    
    for face in faces:
        face = adjust_bounding_box(face, image, f=float(adjust))

        # draw face bouding-box
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 4)

        # detect and draw landmarks
        shape = shape_predictor(image, face)

        for i in range(shape.num_parts):
            point = shape.part(i)
            cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), 1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    plt.show()

    if save:
        cv2.imwrite(f'detection__{os.path.split(image_path)[-1]}', image)


if __name__ == '__main__':
    test_image = os.path.join('WFLW_images', '7--Cheering', '7_Cheering_Cheering_7_16.jpg')

    test_model(model_path='wflw_98_landmarks.dat', image_path=test_image, adjust=0.1)
