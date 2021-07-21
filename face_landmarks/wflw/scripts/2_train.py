import os
import dlib


def get_training_options(tree_depth=4, nu=0.1, cascade_depth=15, verbose=True, 
                         pool_size=1000, num_test_splits=250, oversampling=10, num_threads=8):
    options = dlib.shape_predictor_training_options()
    options.tree_depth = tree_depth
    options.nu = nu
    options.cascade_depth = cascade_depth
    options.feature_pool_size = pool_size
    options.num_test_splits = num_test_splits
    options.oversampling_amount = oversampling
    options.be_verbose = verbose
    options.num_threads = num_threads
    return options


def train_model(name: str, xml: str, **kwargs):
    '''requires: the model name, and the path of the xml annotations.
    It trains and saves a new model according to the specified 
    training options and given annotations'''
    options = get_training_options(**kwargs)

    dlib.train_shape_predictor(xml, name, options)


def measure_model_error(model, xml):
    '''requires: the model and xml path.
    It measures the error of the model on the given
    xml file of annotations.'''
    error = dlib.test_shape_predictor(xml, model)
    print("Error of the model: {} is {}".format(model, error))


if __name__ == '__main__':
	train_model(name='wflw_98_landmarks.dat', 
                xml=os.path.join('WFLW_images', 'labels_train2_WFLW.xml'),
                cascade_depth=10, pool_size=400, num_test_splits=50, oversampling=5)

	# test (on training-set)
	measure_model_error(model='wflw_98_landmarks.dat',
	                    xml=os.path.join('WFLW_images', 'labels_train2_WFLW.xml'))

	# test (on test-set)
	measure_model_error(model='wflw_98_landmarks.dat',
	                    xml=os.path.join('WFLW_images', 'labels_test2_WFLW.xml'))
        