import numpy as np


class CrossEntropyError:
    def __init__(self):
        pass

    def get_loss(self, output_data, answer_label):
        if output_data.ndim == 1:
            output_data = output_data.reshape(1, output_data.size)
            answer_label = answer_label.reshape(1, answer_label.size)
        if answer_label.size == output_data.size:
            answer_label = answer_label.argmax(axis=1)
        batch_size = output_data.shape[0]
        return -np.sum(np.log(output_data[np.arange(batch_size), answer_label] + 1e-7))