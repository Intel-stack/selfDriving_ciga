import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
    model_path = '/home/test/Downloads/model_h5.keras'
    try:
        model = load_model(model_path, compile=False)
    except TypeError:
        # 모델을 재저장하여 호환성을 맞춤
        model = tf.keras.models.load_model(model_path, compile=False)
        new_model_path = '/home/test/Downloads/model_h5_fixed.keras'
        model.save(new_model_path, save_format='h5')
        model = load_model(new_model_path, compile=False)

main()
