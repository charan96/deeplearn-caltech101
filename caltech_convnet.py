from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras import applications
from keras.preprocessing import image
from skimage import io, exposure, color, transform
from sklearn.model_selection import train_test_split
import numpy as np
import cPickle as pickle
import h5py as h5py
import os

# constants and hyperparameters
wrk_dir = '/home/ramcharan/deeplearn/caltech/core'
IMG_SIZE = 200
NUM_CLASSES = 101


def preprocess_img(img):
    try:
        hsv = color.rgb2hsv(img)
    except:
        rgb_img = color.gray2rgb(img)
        hsv = color.rgb2hsv(rgb_img)
        
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    
    min_side = min(img.shape[:-1])
    center = img.shape[0] // 2, img.shape[1] // 2
    img = img[center[0] - min_side // 2:center[0] + min_side // 2,
              center[1] - min_side // 2:center[1] + min_side // 2,
              :]
    
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    
    return img


def build_img_and_classes():
    base_data_dir = '/home/ramcharan/deeplearn/caltech/core/data/categories/'
    
    images = {}
    classes = {}
    
    categories = [categ for categ in os.listdir(base_data_dir)]
    categories.remove('BACKGROUND_Google')
    
    for category in categories:
        for img_file in os.listdir(base_data_dir + category):
            img = preprocess_img(io.imread(base_data_dir + category + '/' + img_file))
            images[category + '/' + img_file] = img
            classes[category + '/' + img_file] = category
            
    with open('img_dict.p', 'wb') as imd, open('class_dict.p', 'wb') as cld:
        pickle.dump(images, imd, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(classes, cld, protocol=pickle.HIGHEST_PROTOCOL)
        
    return images, classes


def load_images_classes():
	if ('img_dict.p' not in os.listdir(wrk_dir)) or ('class_dict.p' not in os.listdir(wrk_dir)):
		images, classes = build_img_and_classes()
	else:
		with open('img_dict.p', 'rb') as imd, open('class_dict.p', 'rb') as cld:
			print 'Loading img_dict.p and classes_dict.p files...'
			images = pickle.load(imd)
			classes = pickle.load(cld)

	return images, classes


def get_train_test_data(images, classes):
	x, y = [], []

	for img_file in images.keys():
		x.append(images[img_file])
		y.append(classes[img_file])
	    
	x = np.array(x, dtype='float32')

	str_uniq_list, int_coded_uniq_list = np.unique(y, return_inverse=True)
	y = np_utils.to_categorical(int_coded_uniq_list, NUM_CLASSES)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

	return x_train, x_test, y_train, y_test


def build_model(x_train, x_test, y_train, y_test):
	# TRANSFER_LEARNING_SETUP
	base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape = (IMG_SIZE, IMG_SIZE, 3))

	for layer in base_model.layers:
		layer.trainable = False
	    
	full_model = Sequential()
	full_model.add(base_model)
	# full_model.add(Dense(512, input_shape=(6, 6, 512), activation='relu'))

	full_model.add(Flatten())
	full_model.add(Dense(256, activation='relu'))
	full_model.add(Dense(256, activation='relu'))
	full_model.add(Dropout(0.5))
	full_model.add(Dense(NUM_CLASSES, activation='softmax'))

	full_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	full_model.fit(x_train, y_train, epochs=20, batch_size=16)


def main():
	images, classes = load_images_classes()
	x_train, x_test, y_train, y_test = get_train_test_data(images, classes)
	
	build_model(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
	main()
