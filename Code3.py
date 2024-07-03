import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def load_images_from_folder(folder, label, image_size):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (image_size, image_size))
            img = img.flatten()  
            images.append(img)
            labels.append(label)
    return images, labels


cat_folder = 'path/to/cats'
dog_folder = 'path/to/dogs'

image_size = 64

cat_images, cat_labels = load_images_from_folder(cat_folder, 0, image_size)
dog_images, dog_labels = load_images_from_folder(dog_folder, 1, image_size)

images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(svm, 'svm_cat_dog_classifier.pkl')

