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

# Set parameters for comparison
CLIENT_NUMBER = 100
CLIENT_RATIO_PER_ROUND = 0.12
EPOCHS = 20  # Reduced number of epochs for comparison
TEST_NUM = 600
FINAL_TEST_NUM = 10000

def evaluate_methods():
    """
    Evaluate all federated learning methods and compare their performance
    """
    print("Starting evaluation of different federated learning methods")
    
    results = {}
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run FedAvg (original implementation in Server.py)
    print("\n===== Running FedAvg =====")
    start_time = time.time()
    
    # Since we can't easily modify the original implementation to store metrics
    # We'll just run it as is and note that it will print its results
    client = Server.buildClients(CLIENT_NUMBER)
    global_vars = client.get_client_vars()
    
    fedavg_acc = []
    fedavg_loss = []
    
    for ep in range(EPOCHS):
        # We are going to sum up active clients' vars at each epoch
        client_vars_sum = None

        # Choose some clients that will train on this epoch
        random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
        total_data_size = sum([client.dataset.train[cid].size
                              for cid in random_clients])

        # Train with these clients
        for client_id in random_clients:
            # Restore global vars to client's model
            client.set_global_vars(global_vars)
             
            # train one client
            data_size = client.train_epoch(cid=client_id)
             
            # obtain current client's vars
            current_client_vars = client.get_client_vars()

            # sum it up with weights
            weight = data_size / total_data_size
            if client_vars_sum is None:
                client_vars_sum = [weight * x for x in current_client_vars]
            else:
                for cv, ccv in zip(client_vars_sum, current_client_vars):
                    cv += weight * ccv

        # assign the avg vars to global vars
        global_vars = client_vars_sum

        # run test and collect metrics
        client.set_global_vars(global_vars)
        acc, loss = client.run_test(TEST_NUM)
        fedavg_acc.append(acc)
        fedavg_loss.append(loss)
        print(f"[epoch {ep + 1}, {TEST_NUM} inst] Testing ACC: {acc:.4f}, Loss: {loss:.4f}")
    
    # Final evaluation
    client.set_global_vars(global_vars)
    final_acc, final_loss = client.run_test(FINAL_TEST_NUM)
    print(f"Final FedAvg results - Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
    
    fedavg_time = time.time() - start_time
    results['FedAvg'] = {
        'accuracy': fedavg_acc,
        'loss': fedavg_loss,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'time': fedavg_time
    }
    
    # Run FedProx
    print("\n===== Running FedProx =====")
    start_time = time.time()
    fedprox_results, fedprox_final_acc, fedprox_final_loss = train_fedprox(
        CLIENT_NUMBER, CLIENT_RATIO_PER_ROUND, EPOCHS, TEST_NUM, mu=0.01)
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
    fedsgd_results, fedsgd_final_acc, fedsgd_final_loss = train_fedsgd(
        CLIENT_NUMBER, CLIENT_RATIO_PER_ROUND, EPOCHS, TEST_NUM)
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
    fednova_results, fednova_final_acc, fednova_final_loss = train_fednova(
        CLIENT_NUMBER, CLIENT_RATIO_PER_ROUND, EPOCHS, TEST_NUM)
    fednova_time = time.time() - start_time
    
    results['FedNova'] = {
        'accuracy': fednova_results['accuracy'],
        'loss': fednova_results['loss'],
        'final_acc': fednova_final_acc,
        'final_loss': fednova_final_loss,
        'time': fednova_time
    }
    
    # Generate plots
    plot_results(results, timestamp)
    
    # Print summary
    print("\n===== Summary =====")
    print(f"{'Method':<10} {'Final Acc':<15} {'Final Loss':<15} {'Time (s)':<10}")
    print('-' * 50)
    
    for method, result in results.items():
        print(f"{method:<10} {result['final_acc']:<15.4f} {result['final_loss']:<15.4f} {result['time']:<10.2f}")
    
    return results

def plot_results(results, timestamp):
    """
    Plot accuracy and loss curves for all methods
    """
    epochs = range(1, EPOCHS + 1)
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    for method, result in results.items():
        plt.plot(epochs, result['accuracy'], label=f"{method} (Final: {result['final_acc']:.4f})")
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/accuracy_comparison_{timestamp}.png")
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    for method, result in results.items():
        plt.plot(epochs, result['loss'], label=f"{method} (Final: {result['final_loss']:.4f})")
    
    plt.title('Model Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/loss_comparison_{timestamp}.png")
    
    # Bar plot for final metrics
    methods = list(results.keys())
    final_accs = [results[m]['final_acc'] for m in methods]
    final_losses = [results[m]['final_loss'] for m in methods]
    training_times = [results[m]['time'] for m in methods]
    
    # Final accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.bar(methods, final_accs)
    plt.title('Final Accuracy')
    plt.ylim(0, 1)
    
    # Final loss comparison
    plt.subplot(1, 3, 2)
    plt.bar(methods, final_losses)
    plt.title('Final Loss')
    
    # Training time comparison
    plt.subplot(1, 3, 3)
    plt.bar(methods, training_times)
    plt.title('Training Time (s)')
    
    plt.tight_layout()
    plt.savefig(f"results/final_metrics_comparison_{timestamp}.png")

if __name__ == "__main__":
    evaluate_methods() 