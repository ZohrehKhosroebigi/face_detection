import numpy as np
from keras.preprocessing.image import load_img
class User_page():
    @property
    def user_page(self):
        user_img = input("Past your image on test folder and enter its name:  ")
        self.img_path = 'test/' + user_img
        resized_img = load_img(self.img_path, target_size=(96, 96))
        self.img_path='test/' +'img_temp.jpg'
        resized_img.save(self.img_path)
        return self.img_path