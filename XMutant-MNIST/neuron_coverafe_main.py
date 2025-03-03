from utils import set_all_seeds
from data_anaylsis.neuron_coverage import model_neuron_coverage
from population import load_mnist_test
from config import POPSIZE
from predictor import Predictor

pop_size = POPSIZE

for digit in range(10): # range(10):
    set_all_seeds(digit)

    x_test, y_test = load_mnist_test(pop_size, digit)
    model = Predictor.model
    coverage = model_neuron_coverage(model, x_test, threshold=0.5)
    print(f"Digit {digit} Original Test Set Model Neuron Coverage: {coverage:.2f}")



