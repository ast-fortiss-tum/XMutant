from utils import set_all_seeds, load_mnist_test

from config import POPSIZE
from predictor import Predictor
import os
import csv
from adversarial_methods.FGSM.fgsm import *
from PIL import Image
pop_size = POPSIZE
model = Predictor.model

def generate_one_example(image,image_label, model):
    image = np.reshape(image, model.input_shape[1:])
    x = np.array([image])
    y = np.zeros((1,10))
    y[0,image_label] = 1
    g = gradient_of_x(x, y, model)
    g_npy = np.squeeze(g.numpy())
    g_sign = np.reshape(tf.sign(g_npy), model.input_shape[1:])

    epsilons = np.linspace(0, 0.5, num=10)

    for eps in epsilons:
        adv_x = image + eps*g_sign
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        prediction, confidence = Predictor.predict_single(np.expand_dims(adv_x, 0), image_label)
        if prediction!= image_label:
            print(f"eps {eps} misclass, label {image_label}, predict {prediction, confidence}")
            return True, {'adv_x': adv_x, 'prediction': prediction, 'confidence': confidence, 'eps': eps}
    return False, None

def main():
    img_dir = "./result/fgsm_digits/"
    os.makedirs(img_dir, exist_ok=True)
    csv_file = "./result/fgsm_digits/record_FGSM.csv"
    for digit in range(10): # range(10):
        set_all_seeds(digit)
        x_test, y_test = load_mnist_test(pop_size, digit)
        for i, (image,image_label) in enumerate(zip(x_test, y_test)):
            image = image.astype('float32')/255 # normalization
            succeed, result = generate_one_example(image, image_label,model)
            if succeed:
                # save image
                save_path = os.path.join(img_dir, f"digit_{digit}")
                os.makedirs(save_path, exist_ok=True)
                img_name = os.path.join(save_path, f"id_{i}.png")

                img_array = result['adv_x'].numpy().reshape(28,28)
                img = Image.fromarray(np.uint8(img_array*255))
                img.save(img_name)
                # record csv
                if os.path.exists(csv_file):
                    with open(csv_file, 'a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([digit,
                                             i,
                                             result['eps'],
                                             image_label,
                                             result['prediction'],
                                             result['confidence'],
                                             img_name])
                else:
                    with open(csv_file, 'w', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([
                            'digit',
                            'id',
                            'eps',
                            'expected_label',
                            'predicted_label',
                            'confidence',
                            'image_path'
                        ])
                        csv_writer.writerow([digit,
                                             i,
                                             result['eps'],
                                             image_label,
                                             result['prediction'],
                                             result['confidence'],
                                             img_name])








if __name__=="__main__":

    main()