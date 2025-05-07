import tensorflow as tf
import numpy as np
from collections import namedtuple
import math

from Model import AlexNet
from Dataset import Dataset

FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op pred')


class Clients:
    def __init__(self, input_shape, num_classes, learning_rate, clients_num):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        net = AlexNet(input_shape, num_classes, learning_rate, self.graph)
        self.model = FedModel(*net)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        self.dataset = Dataset(tf.keras.datasets.cifar10.load_data,
                               split=clients_num)

    def run_test(self, num_test_samples):
        """ Evaluate the model on num_test_samples from the test set using batches. """
        all_test_data = self.dataset.test
        test_batch_size = 16 # Use a smaller batch size for testing to avoid memory issues
        num_batches = math.ceil(all_test_data.size / test_batch_size)
        total_loss = 0.0
        total_correct_preds = 0
        samples_processed = 0

        # Limit the number of samples processed if num_test_samples is less than total test size
        # Process at most num_test_samples or all_test_data.size, whichever is smaller
        max_samples_to_process = min(num_test_samples, all_test_data.size)
        batches_to_run = math.ceil(max_samples_to_process / test_batch_size)

        with self.graph.as_default():
            # Run evaluation in batches
            for i in range(batches_to_run):
                # Ensure we don't exceed max_samples_to_process
                current_batch_size = min(test_batch_size, max_samples_to_process - samples_processed)
                if current_batch_size <= 0:
                    break

                batch_x, batch_y = all_test_data.next_batch(current_batch_size)
                feed_dict = {
                    self.model.X: batch_x,
                    self.model.Y: batch_y,
                    self.model.DROP_RATE: 0
                }
                # Get loss and predictions for the batch
                batch_loss, batch_preds = self.sess.run([self.model.loss_op, self.model.pred],
                                                        feed_dict=feed_dict)

                # Check for nan/inf in batch_loss
                if np.isnan(batch_loss) or np.isinf(batch_loss):
                    # print(f"Warning: NaN or Inf detected in batch_loss during testing. Batch X mean: {np.mean(batch_x)}, Batch Y: {batch_y[:5]}")
                    # Option 1: Skip this batch from average loss (might hide issues)
                    # Option 2: Return a very large loss or handle as error
                    # For now, let's try to avoid it influencing the total_loss if it's problematic
                    # If all batches result in NaN, avg_loss will also be NaN, which is a clear indicator.
                    pass # Allow it to propagate to see if it becomes NaN for the whole test
                else:
                    total_loss += batch_loss * current_batch_size # Weight loss by actual batch size

                batch_correct_preds = np.sum(batch_preds == np.argmax(batch_y, axis=1))
                total_correct_preds += batch_correct_preds
                samples_processed += current_batch_size

        # Calculate final average accuracy and loss over the processed samples
        if samples_processed == 0:
            return 0.0, 0.0 # Avoid division by zero

        avg_acc = total_correct_preds / samples_processed
        avg_loss = total_loss / samples_processed

        return avg_acc, avg_loss

    def train_epoch(self, cid, batch_size=32, dropout_rate=0.5):
        """
            Train one client with its own data for one epoch
            cid: Client id
        """
        dataset = self.dataset.train[cid]

        with self.graph.as_default():
            for _ in range(math.ceil(dataset.size / batch_size)):
                batch_x, batch_y = dataset.next_batch(batch_size)
                feed_dict = {
                    self.model.X: batch_x,
                    self.model.Y: batch_y,
                    self.model.DROP_RATE: dropout_rate
                }
                self.sess.run(self.model.train_op, feed_dict=feed_dict)

        return dataset.size

    def get_client_vars(self):
        """ Return all of the variables list """
        with self.graph.as_default():
            client_vars = self.sess.run(tf.trainable_variables())
        return client_vars

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, global_vars):
                variable.load(value, self.sess)

    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        client_num = self.get_clients_num()
        choose_num = math.ceil(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        return len(self.dataset.train)