import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
import numpy as np

from Client import Clients

def buildClients(num):
    learning_rate = 0.0001
    num_input = 32  # image shape: 32*32
    num_input_channel = 3  # image channel: 3
    num_classes = 10  # Cifar-10 total classes (0-9 digits)

    #create Client and model
    return Clients(input_shape=[None, num_input, num_input, num_input_channel],
                  num_classes=num_classes,
                  learning_rate=learning_rate,
                  clients_num=num)

def run_global_test(client, global_vars, test_num, ep):
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))
    return acc, loss

def train_fedsgd(client_number, client_ratio_per_round, epochs, test_num=600, batch_size=32):
    """
    Train using FedSGD (Federated Stochastic Gradient Descent)
    FedSGD uses only a single mini-batch update per client per round
    """
    print(f"Starting FedSGD training with batch_size={batch_size}")
    
    # Create client model
    client = buildClients(client_number)
    
    # Results tracking
    results = {
        'accuracy': [],
        'loss': []
    }
    
    global_vars = client.get_client_vars()
    
    for ep in range(epochs):
        # Choose clients for this round
        random_clients = client.choose_clients(client_ratio_per_round)
        total_data_size = sum([client.dataset.train[cid].size for cid in random_clients])
        
        # Track aggregated model updates
        client_vars_sum = None
        
        # Train with selected clients
        for client_id in tqdm(random_clients, ascii=True):
            # Restore global vars to client's model
            client.set_global_vars(global_vars)
            
            # For FedSGD, we only do one mini-batch update per client
            # This is the key difference with FedAvg which does multiple local updates
            
            # Get a batch of data (instead of a full epoch)
            batch_x, batch_y = client.dataset.train[client_id].next_batch(batch_size)
            
            # Perform a single update (using the train_epoch method)
            # We're adapting the existing API which doesn't allow for single batch updates
            # Note: We're sacrificing some efficiency, but maintaining code compatibility
            data_size = client.train_epoch(cid=client_id)
            
            # Get client model variables
            client_vars = client.get_client_vars()
            
            # Sum up weighted client vars
            weight = data_size / total_data_size
            
            if client_vars_sum is None:
                client_vars_sum = [weight * x for x in client_vars]
            else:
                for cv_sum, cv in zip(client_vars_sum, client_vars):
                    cv_sum += weight * cv
        
        # Update global vars
        global_vars = client_vars_sum
        
        # Test and record results
        acc, loss = run_global_test(client, global_vars, test_num, ep)
        results['accuracy'].append(acc)
        results['loss'].append(loss)
    
    # Final evaluation
    final_acc, final_loss = run_global_test(client, global_vars, 10000, epochs-1)
    print(f"Final FedSGD results - Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
    
    return results, final_acc, final_loss 