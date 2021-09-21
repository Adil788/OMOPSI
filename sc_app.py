from fastai.vision.widgets import *
from fastai.vision.all import *
from pathlib import Path
import streamlit as st
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class SceneClassifier:
    def __init__(self, filename):
        self.classifier_model = load_learner(filename)
        self.img = self.get_uploaded_img()
        if self.img is not None:
            self.display_output()
            self.get_prediction()

    @staticmethod
    def get_uploaded_img():
        sc_file = st.file_uploader("Select Image", type=['png', 'jpeg', 'jpg'])
        if sc_file is not None:
            return PILImage.create((sc_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500, 500), caption='Selected Image')

    def get_prediction(self):

        if st.button('Classify Selected Scene'):
            pred, idx, probs = self.classifier_model.predict(self.img)
            st.write(f'Class: {pred}; Probability: {probs[idx]:.03f}')
        else:
            st.write(f'Select and Classify Image')


if __name__ == '__main__':
    model_path = './model1.pkl'
    predictor = SceneClassifier(model_path)
