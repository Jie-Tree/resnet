from keras.models import load_model
from PIL import Image
import numpy as np
from managedata import num2lable

class Predict():
    def __init__(self):
        self.model = load_model('.h5')

    def read_image(self, img_name):
        im = Image.open(img_name)
        data = np.array(im)
        data = np.reshape(data, [1, 512, 512, 1])
        return data

    def predict(self, img):
        x = self.read_image(img)
        x = x.astype('float32')
        mean_image = np.mean(x, axis=0)
        x -= mean_image
        x /= 128.
        return self.model.predict(x)


if __name__ == '__main__':
    p = Predict()
    y = p.predict('TrainingData/0322.jpg')
    num = np.argmax(y)
    print(y)
    print(num)
    print(num2lable(num))
