from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K 
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import cv2
import os

# Set the path for saving the model
model_path = 'model.h5'

root_dir = 'train'

images = []
labels = []

# Load images and labels
for emotion in os.listdir(root_dir):
    emotion_dir = os.path.join(root_dir, emotion)
    
    if os.path.isdir(emotion_dir):
        for file in os.listdir(emotion_dir):
            file_path = os.path.join(emotion_dir, file)
            
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (32, 32))
                
                images.append(img)
                labels.append(emotion)

# Convert to numpy arrays
x = np.array(images)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Reshape images
x_reshaped = x.reshape(-1, 32, 32, 1)
x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y_onehot)

# Normalize the data
train_data = x_train.astype("float32") / 255.0
test_data = x_test.astype("float32") / 255.0

# Compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=0.001)
model = LeNet.build(numChannels=1, imgRows=32, imgCols=32, numClasses=7)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.summary()

# Train the model
print("[INFO] training...")
model.fit(train_data, y_train, epochs=20, verbose=1)

# Evaluate the model
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(test_data, y_test, batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# Save the entire model to H5 file
print("[INFO] saving model to file...")
model.save(model_path)

# Prediction on random samples
for i in np.random.choice(np.arange(0, len(y_test)), size=(10,)):
    # classify the digit
    probs = model.predict(test_data[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # extract the image from the testData
    if K.image_data_format() == "channels_first":
        image = (test_data[i][0] * 255).astype("uint8")
    else:
        image = (test_data[i] * 255).astype("uint8")

    # merge the channels into one image
    image = cv2.merge([image] * 3)
    # resize the image for better visibility
    image = cv2.resize(image, (200, 150), interpolation=cv2.INTER_LINEAR)
    
    # Get the predicted class name and actual class name
    predicted_class_name = encoder.inverse_transform(prediction)[0]
    actual_class_name = encoder.inverse_transform([np.argmax(y_test[i])])[0]

    # show the image and prediction
    cv2.putText(image, f"Pred: {predicted_class_name}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(predicted_class_name, actual_class_name))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)
