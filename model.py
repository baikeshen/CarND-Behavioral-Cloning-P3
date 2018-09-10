import numpy as np
import csv
import cv2
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D

# Constants
data_path = "data/"
image_path = data_path + "IMG/"
left_image_angle_correction = 0.20
right_image_angle_correction = -0.20
csv_data = []
processed_csv_data = []


# Method to pre-process the input image
def pre_process_image(image):
    # Since cv2 reads the image in BGR format and the simulator will send the image in RGB format
    # Hence changing the image color space from BGR to RGB
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Cropping the image
    #cropped_image = colored_image[60:140, :]
    # Downscaling the cropped image
    #resized_image = cv2.resize(cropped_image, None, fx=0.25, fy=0.4, interpolation=cv2.INTER_CUBIC)
    return colored_image #cropped_image

# Reading the content of csv file
with open(data_path + 'driving_log.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skipping the headers
    next(csv_reader, None)
    for each_line in csv_reader:
        csv_data.append(each_line)

# Getting shape of processed image
first_img_path = image_path + csv_data[0][0].split('/')[-1]
first_image = cv2.imread(first_img_path)
processed_image_shape = pre_process_image(first_image).shape

print(processed_image_shape)


images = []
measurements = []

for Line in csv_data:
    current_image_path = image_path + Line[0].split('/')[-1]
    current_image = cv2.imread(current_image_path)
    current_image = pre_process_image(current_image)
    images.append(current_image)

    measurement = float(Line[3])
    measurements.append(measurement)
  #------------------------------------------------
    current_image_path = image_path + Line[1].split('/')[-1]
    current_image = cv2.imread(current_image_path)
    current_image = pre_process_image(current_image)
    images.append(current_image)

    measurement = float(Line[3]) + 0.2
    measurements.append(measurement)
  #-------------------------------------------------
    current_image_path = image_path + Line[2].split('/')[-1]
    current_image = cv2.imread(current_image_path)
    current_image = pre_process_image(current_image)
    images.append(current_image)

    measurement = float(Line[3]) - 0.2
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#X_train = np.array(images)
#y_train = np.array(measurements)


# My final model architecture
model = Sequential()


model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=processed_image_shape)) #(160, 320, 3)


model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
#model.add(MaxPooling2D())

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
#model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(100))  #100
#model.add(Dropout(0.20))
model.add(Dense(50))   # 50
model.add(Dense(10))   # 10
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 3)

model.save('model_nvidia.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

exit()
