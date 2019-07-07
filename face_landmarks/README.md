# Minified Face Landmarks Models
Detect 68 face landmark points with the **face_landmarks_68.dat** model in just 31 MB! The other models are able to detect specifics set of landmarks.

I've provided an example **training script**, so you can build your own models with custom parameters (like the ones below) and a custom set of facial landmarks.


## Specific Models
If you don't need to detect all the 68 landmarks, you can use:

- The __eye_eyebrows_22.dat__ to detect eyes and eyebrows (10.5 MB)
  ![eyes and eyebrows landmarks](https://github.com/Luca96/dlib-minified-models/blob/master/face_landmarks/images/eye_eyebrows.jpg)
- The __nose_mouth_30.dat__ to detect nose and mouth (13.7 MB)
![node and mouth landmarks](https://github.com/Luca96/dlib-minified-models/blob/master/face_landmarks/images/nose_mouth.jpg)
- The __face_contour_17.dat__ to detect the face contour (8.30 MB)
![face contour landmarks](https://github.com/Luca96/dlib-minified-models/blob/master/face_landmarks/images/face_contour.jpg)

## Training Parameters
Here are the parameter used to train the various models (these are a good tradeoff between size, speed and accuracy):
```python
options = dlib.shape_predictor_training_options()
options.tree_depth = 3  # keep to 3 for a smaller model size (otherwise use 4)
options.nu = 0.1
options.cascade_depth = 10  # increase to obtain more accurate results
options.feature_pool_size = 150  # make the model execute faster
options.num_test_splits = 350  # choose better feature, during training
```

If you don't care about `size`, `speed` but you want the maximum __accuracy__, try these:
```python
options = dlib.shape_predictor_training_options()
options.tree_depth = 4
options.nu = 0.1
options.cascade_depth = 15
options.feature_pool_size = 800  # or even 1000
options.num_test_splits = 200  # 150-200 is enough
```

## Accuracy

The table below shows the training and test accuracy of the model. The accuracy is computed from the **iBug-300W** dataset, used to train all the models.

|           | FL68  | EE22 | NM30 | FC17  |
| --------- | ----- | ---- | ---- | ----- |
| Training  | 9.44  | 7.30 | 6.54 | 15.30 |
| Test      | 10.89 | 8.40 | 7.86 | 20.13 |
| Size (Mb) | 31.6  | 10.5 | 13.7 | 8.3   |

