# Widya Sari Wibowo (21091397070)

# insialisasi numpys
import numpy as np

# inisialisasi variable
# Input layer feature 10
# Per batchnya 6 input
inputs = [[1.0, 1.5, 2.0, 2.5, 3.3, 3.5, 4.9, 4.5, 5.0, 5.5],
          [1.7, 1.6, 2.4, 2.8, 3.4, 3.6, 4.4, 4.8, 5.2, 5.4],
          [9.2, 4.2, 1.3, 8.2, 2.4, 8.4, 5.8, 7.4, 1.6, 9.3],
          [2.8, 1.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
          [2.5, 6.4, 7.2, 7.4, 8.2, 8.4, 7.2, 9.4, 1.2, 3.4],
          [2.3, 5.4, 2.4, 3.2, 3.4, 4.2, 6.4, 7.7, 8.2, 2.5]]

# inisialisasi bobot variable
# jumlah weight sesuai dengan jumlah neuron layer1, yaitu 5
weights1 = [[0.5, 0.7, 1.4, 2.7, 7.8, 9.4, 3.2, 4.6, 7.3, 9.4],
           [1.5, 3.4, 0.9, 3.2, 0.4, 0.1, 2.8, 6.2, 8.4, 3.7],
           [2.7, 1.3, 1.4, 7.2, 9.8, 0.2, 6.5, 8.4, 5.3, 6.4],
           [6.1, 9.3, 4.2, 7.4, 0.3, 2.5, 1.3, 9.3, 8.2, 4.5],
           [2.7, 8.5, 0.2, 1.5, 3.2, 1.9, 0.8, 4.3, 6.4, 4.8]]

# inisialisasi bias
# jumlah bias pada layer1, yaitu 5
bias1 = [2.3, 3.5, 0.1, 1.5, 3.9]

# panjang weights sesuai dengan neuron layer1, yaitu 5
# jumlah weights sesuai dengan jumlah neuron layer2, yaitu 3
weights2 = [[2.4, 7.3, 4.9, 1.3, 8.3],
            [3.5, 2.6, 1.7, 6.4, 7.5],
            [1.6, 7.9, 6.3, 2.6, 8.4]]

# jumlah bias pada layer2, yaitu 3 neuron
bias2 = [3.1, 1.2, 4.5]

# command untuk menghitung layer1 menggunakan inputs, weights1, dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1) . T) + bias1

# command untuk menghitung layer2 menggunakan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2) . T) + bias2

# mencetak output layer2
print(layer2_outputs)