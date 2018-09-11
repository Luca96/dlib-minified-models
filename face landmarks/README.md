# Minified Face Landmarks Models
Detect 68 face landmark points with the **face_landmarks_68.dat** model in just 31 MB!


## Specific Models
If you don't need to detect all the 68 landmarks, you can use:

- The __eye_eyebrows_22.dat__ to detect eyes and eyebrows (10.5 MB)
  ![eyes and eyebrows landmarks](https://github.com/Luca96/dlib-minified-models/blob/master/face%20landmarks/images/eye_eyebrows.jpg)
- The __nose_mouth_30.dat__ to detect nose and mouth (13.7 MB)
![node and mouth landmarks](https://github.com/Luca96/dlib-minified-models/blob/master/face%20landmarks/images/nose_mouth.jpg)
- The __face_contour_17.dat__ to detect the face contour (9.13 MB)
![face contour landmarks](https://github.com/Luca96/dlib-minified-models/blob/master/face%20landmarks/images/face_contour.jpg)

## Training Parameters:
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
