import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
import math
from tqdm import tqdm

from Model import AlexNet
from Dataset import Dataset

def train_centralized(epochs, learning_rate=0.0001, batch_size=32, test_num=600, final_test_num=10000):
    """
    Train the AlexNet model using standard centralized training.

    Args:
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Training batch size.
        test_num: Number of instances for intermediate testing.
        final_test_num: Number of instances for final testing.

    Returns:
        A dictionary containing training history and final metrics.
    """
    print("===== Starting Centralized Training =====")
    start_time = time.time()

    # 1. Load Dataset (centrally, split=0)
    # Ensure dataset path or loading mechanism is correct
    try:
        cifar10_dataset = tf.keras.datasets.cifar10.load_data
    except AttributeError:
        print("Error: Could not find tf.keras.datasets.cifar10.load_data")
        print("Make sure TensorFlow is installed correctly.")
        return None, None, None # Indicate failure

    dataset = Dataset(cifar10_dataset, split=0) # split=0 for centralized data
    train_data = dataset.train
    test_data = dataset.test

    # 2. Define Model Parameters
    num_input = 32  # image shape: 32*32
    num_input_channel = 3  # image channel: 3
    num_classes = 10  # Cifar-10 total classes
    input_shape = [None, num_input, num_input, num_input_channel]

    # 3. Build Model
    graph = tf.Graph()
    with graph.as_default():
        # Use the same AlexNet model definition
        X, Y, DROP_RATE, train_op, loss_op, acc_op, pred = AlexNet(input_shape, num_classes, learning_rate, graph)
        # Initializer
        init = tf.global_variables_initializer()

    # 4. Training Session
    sess = tf.Session(graph=graph)
    sess.run(init)

    # Results tracking
    results_history = {
        'accuracy': [],
        'loss': []
    }

    print(f"Training for {epochs} epochs with batch size {batch_size}...")

    # 5. Training Loop
    for ep in range(epochs):
        num_batches = math.ceil(train_data.size / batch_size)
        pbar = tqdm(range(num_batches), desc=f"Epoch {ep+1}/{epochs}")
        epoch_loss = 0.0
        for _ in pbar:
            batch_x, batch_y = train_data.next_batch(batch_size)
            feed_dict = {
                X: batch_x,
                Y: batch_y,
                DROP_RATE: 0.5 # Use dropout during training
            }
            _, current_loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
            epoch_loss += current_loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        # Evaluate after each epoch
        test_x, test_y = test_data.next_batch(test_num) # Use specified test_num
        feed_dict_test = {
            X: test_x,
            Y: test_y,
            DROP_RATE: 0 # No dropout during testing
        }
        acc, loss = sess.run([acc_op, loss_op], feed_dict=feed_dict_test)

        results_history['accuracy'].append(acc)
        results_history['loss'].append(loss)
        avg_epoch_loss = epoch_loss / num_batches
        print(f"[Epoch {ep + 1}/{epochs}, {test_num} inst] Testing ACC: {acc:.4f}, Loss: {loss:.4f} (Avg Train Loss: {avg_epoch_loss:.4f})")

    # 6. Final Evaluation (in batches to avoid memory issues)
    print(f"\nPerforming final evaluation on {test_data.size} test instances...")
    final_total_loss = 0.0
    final_correct_preds = 0
    final_eval_batch_size = 16 # Use a smaller batch size for final evaluation
    final_batches = math.ceil(test_data.size / final_eval_batch_size)

    for _ in tqdm(range(final_batches), desc="Final Evaluation"):
        # Use final_eval_batch_size here
        final_test_x, final_test_y = test_data.next_batch(final_eval_batch_size)
        feed_dict_final_test = {
            X: final_test_x,
            Y: final_test_y,
            DROP_RATE: 0
        }
        # Get loss and predictions for the batch
        batch_loss, batch_preds = sess.run([loss_op, pred], feed_dict=feed_dict_final_test)
        final_total_loss += batch_loss * len(final_test_x) # Weight loss by batch size
        # Calculate correct predictions for the batch
        batch_correct_preds = np.sum(batch_preds == np.argmax(final_test_y, axis=1))
        final_correct_preds += batch_correct_preds

    # Calculate final average accuracy and loss
    final_acc = final_correct_preds / test_data.size
    final_loss = final_total_loss / test_data.size

    print(f"Final Centralized results - Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")

    # 7. Cleanup and Return
    sess.close()
    total_time = time.time() - start_time
    print(f"Total Centralized Training Time: {total_time:.2f} seconds")

    return results_history, final_acc, final_loss, total_time

# Example of how to run (optional, for testing this script directly)
if __name__ == "__main__":
    EPOCHS = 20 # Example epochs
    results, final_acc, final_loss, total_time = train_centralized(epochs=EPOCHS)

    if results:
        # Plot if needed
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, EPOCHS + 1), results['accuracy'])
        plt.title('Centralized Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, EPOCHS + 1), results['loss'])
        plt.title('Centralized Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show() 