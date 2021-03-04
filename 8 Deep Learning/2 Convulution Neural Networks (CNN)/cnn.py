# no preprocessing required , image preprocessing will b e done later

# model preparation

# importing all the required libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Flatten

# step 0 : initializing the cnn model
classifier = Sequential()

# step 1 : add convolution later
classifier.add( Conv2D( 32, (3,3), activation='relu', input_shape = (64,64,3) ) )

# step 2: addd maxpooling later
classifier.add( MaxPooling2D( pool_size=(2,2)))

# step 3: Flattening
classifier.add( Flatten() )

# step 4: full connection
classifier.add( Dense( units = 128, activation= 'relu'))
classifier.add( Dense( units = 1, activation= 'sigmoid' ))

# compiling the cnn model
classifier.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the model to the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 2000)





















