import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime

# Import the different FL methods
import Server  # Original FedAvg implementation
from FedProx import train_fedprox
from FedSGD import train_fedsgd
from FedNova import train_fednova
# Import Centralized Training
from CentralizedTraining import train_centralized

# Set parameters for comparison
CLIENT_NUMBER = 100
CLIENT_RATIO_PER_ROUND = 0.12
EPOCHS = 20  # Reduced number of epochs for comparison
TEST_NUM = 600
FINAL_TEST_NUM = 10000
LEARNING_RATE = 0.0001 # Define learning rate here for consistency
BATCH_SIZE = 32 # Define batch size for centralized training

def evaluate_methods():
    """
    Evaluate all federated learning methods and centralized training, comparing their performance
    """
    print("Starting evaluation of different training methods")

    results = {}

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run Centralized Training
    print("\n===== Running Centralized Training =====")
    centralized_history, centralized_final_acc, centralized_final_loss, centralized_time = train_centralized(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        test_num=TEST_NUM,
        final_test_num=FINAL_TEST_NUM
    )
    if centralized_history: # Check if training was successful
        results['Centralized'] = {
            'accuracy': centralized_history['accuracy'],
            'loss': centralized_history['loss'],
            'final_acc': centralized_final_acc,
            'final_loss': centralized_final_loss,
            'time': centralized_time
        }
    else:
        print("Centralized training failed, skipping...")

    # Run FedAvg (original implementation in Server.py)
    print("\n===== Running FedAvg =====")
    start_time = time.time()

    # Build client with consistent learning rate
    client_fedavg = Server.buildClients(CLIENT_NUMBER) # Learning rate is set inside buildClients
    global_vars_fedavg = client_fedavg.get_client_vars()

    fedavg_acc = []
    fedavg_loss = []

    for ep in range(EPOCHS):
        client_vars_sum = None
        random_clients = client_fedavg.choose_clients(CLIENT_RATIO_PER_ROUND)
        total_data_size = sum([client_fedavg.dataset.train[cid].size
                              for cid in random_clients])

        # We need a way to pass batch_size to train_epoch if it's used there
        # Assuming train_epoch uses a default or fixed batch size internally based on Client.py
        # Current Client.py train_epoch has batch_size parameter, let's use BATCH_SIZE
        for client_id in random_clients:
            client_fedavg.set_global_vars(global_vars_fedavg)
            # Pass BATCH_SIZE to train_epoch
            data_size = client_fedavg.train_epoch(cid=client_id, batch_size=BATCH_SIZE)
            current_client_vars = client_fedavg.get_client_vars()
            weight = data_size / total_data_size
            if client_vars_sum is None:
                client_vars_sum = [weight * x for x in current_client_vars]
            else:
                for i in range(len(client_vars_sum)):
                     client_vars_sum[i] += weight * current_client_vars[i]

        global_vars_fedavg = client_vars_sum
        client_fedavg.set_global_vars(global_vars_fedavg)
        acc, loss = client_fedavg.run_test(TEST_NUM)
        fedavg_acc.append(acc)
        fedavg_loss.append(loss)
        print(f"[Epoch {ep + 1}, {TEST_NUM} inst] Testing ACC: {acc:.4f}, Loss: {loss:.4f}")

    client_fedavg.set_global_vars(global_vars_fedavg)
    final_acc_fedavg, final_loss_fedavg = client_fedavg.run_test(FINAL_TEST_NUM)
    print(f"Final FedAvg results - Accuracy: {final_acc_fedavg:.4f}, Loss: {final_loss_fedavg:.4f}")

    fedavg_time = time.time() - start_time
    results['FedAvg'] = {
        'accuracy': fedavg_acc,
        'loss': fedavg_loss,
        'final_acc': final_acc_fedavg,
        'final_loss': final_loss_fedavg,
        'time': fedavg_time
    }

    # Run FedProx
    print("\n===== Running FedProx =====")
    start_time = time.time()
    # Pass BATCH_SIZE to train_fedprox
    fedprox_results, fedprox_final_acc, fedprox_final_loss = train_fedprox(
        CLIENT_NUMBER, CLIENT_RATIO_PER_ROUND, EPOCHS, TEST_NUM, mu=0.01, batch_size=BATCH_SIZE)
    fedprox_time = time.time() - start_time

    results['FedProx'] = {
        'accuracy': fedprox_results['accuracy'],
        'loss': fedprox_results['loss'],
        'final_acc': fedprox_final_acc,
        'final_loss': fedprox_final_loss,
        'time': fedprox_time
    }

    # Run FedSGD
    print("\n===== Running FedSGD =====")
    start_time = time.time()
    # Pass BATCH_SIZE (already done)
    fedsgd_results, fedsgd_final_acc, fedsgd_final_loss = train_fedsgd(
        CLIENT_NUMBER, CLIENT_RATIO_PER_ROUND, EPOCHS, TEST_NUM, batch_size=BATCH_SIZE)
    fedsgd_time = time.time() - start_time

    results['FedSGD'] = {
        'accuracy': fedsgd_results['accuracy'],
        'loss': fedsgd_results['loss'],
        'final_acc': fedsgd_final_acc,
        'final_loss': fedsgd_final_loss,
        'time': fedsgd_time
    }

    # Run FedNova
    print("\n===== Running FedNova =====")
    start_time = time.time()
    # Pass BATCH_SIZE to train_fednova
    fednova_results, fednova_final_acc, fednova_final_loss = train_fednova(
        CLIENT_NUMBER, CLIENT_RATIO_PER_ROUND, EPOCHS, TEST_NUM, batch_size=BATCH_SIZE)
    fednova_time = time.time() - start_time

    results['FedNova'] = {
        'accuracy': fednova_results['accuracy'],
        'loss': fednova_results['loss'],
        'final_acc': fednova_final_acc,
        'final_loss': fednova_final_loss,
        'time': fednova_time
    }

    # --- DIAGNOSTIC PRINT --- #
    print("\n===== Collected Results Dictionary =====")
    import pprint
    pprint.pprint(results)
    print("====================================\n")
    print("DEBUG: Finished printing results dictionary.") # DEBUG PRINT
    # --- END DIAGNOSTIC PRINT --- #

    # Generate plots
    print("DEBUG: About to generate plots...") # DEBUG PRINT
    # plot_results(results, timestamp, results_dir) # Temporarily commented out
    print("DEBUG: Finished generating plots (or skipped). ") # DEBUG PRINT

    # Print summary
    print("DEBUG: About to print summary table...") # DEBUG PRINT
    print("\n===== Summary =====")
    # Adjust formatting for potentially longer method names
    print(f"{'Method':<12} {'Final Acc':<15} {'Final Loss':<15} {'Time (s)':<10}")
    print('-' * 55)

    # Sort results for consistent table order (optional)
    # sorted_methods = sorted(results.keys())
    # for method in sorted_methods:
    for method, result in results.items():
        print(f"{method:<12} {result['final_acc']:<15.4f} {result['final_loss']:<15.4f} {result['time']:<10.2f}")

    print("DEBUG: Finished printing summary table.") # DEBUG PRINT
    return results

def plot_results(results, timestamp, results_dir):
    """
    Plot accuracy and loss curves for all methods, including centralized.
    """
    if not results: # Check if results dictionary is empty
        print("No results to plot.")
        return

    # Determine the number of epochs based on the first available result
    # Assumes all methods run for the same number of epochs
    first_method = next(iter(results))
    num_epochs_actual = len(results[first_method]['accuracy'])
    epochs_range = range(1, num_epochs_actual + 1)

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    for method, result in results.items():
        # Ensure the data length matches the determined number of epochs
        acc_data = result['accuracy'][:num_epochs_actual]
        plt.plot(epochs_range, acc_data, label=f"{method} (Final: {result['final_acc']:.4f})")

    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epochs / Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"accuracy_comparison_{timestamp}.png"))
    plt.close() # Close the figure to free memory

    # Loss plot
    plt.figure(figsize=(10, 6))
    for method, result in results.items():
        loss_data = result['loss'][:num_epochs_actual]
        plt.plot(epochs_range, loss_data, label=f"{method} (Final: {result['final_loss']:.4f})")

    plt.title('Model Loss Comparison')
    plt.xlabel('Epochs / Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_comparison_{timestamp}.png"))
    plt.close()

    # Bar plot for final metrics
    methods = list(results.keys())
    final_accs = [results[m]['final_acc'] for m in methods]
    final_losses = [results[m]['final_loss'] for m in methods]
    training_times = [results[m]['time'] for m in methods]

    # Adjust figure size based on number of methods
    num_methods = len(methods)
    bar_fig_width = max(8, num_methods * 1.5) # Adjust width dynamically
    plt.figure(figsize=(bar_fig_width, 5))

    # Final accuracy comparison
    plt.subplot(1, 3, 1)
    plt.bar(methods, final_accs, color=plt.cm.viridis(np.linspace(0, 1, num_methods)))
    plt.title('Final Accuracy')
    plt.ylim(bottom=0) # Start y-axis at 0
    plt.xticks(rotation=45, ha='right')

    # Final loss comparison
    plt.subplot(1, 3, 2)
    plt.bar(methods, final_losses, color=plt.cm.viridis(np.linspace(0, 1, num_methods)))
    plt.title('Final Loss')
    plt.ylim(bottom=0)
    plt.xticks(rotation=45, ha='right')

    # Training time comparison
    plt.subplot(1, 3, 3)
    plt.bar(methods, training_times, color=plt.cm.viridis(np.linspace(0, 1, num_methods)))
    plt.title('Training Time (s)')
    plt.ylim(bottom=0)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"final_metrics_comparison_{timestamp}.png"))
    plt.close()

if __name__ == "__main__":
    final_results = evaluate_methods()
    # You can do further analysis with final_results if needed
    # print(final_results) 