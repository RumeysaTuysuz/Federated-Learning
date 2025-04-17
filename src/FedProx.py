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

def train_fedprox(client_number, client_ratio_per_round, epochs, test_num=600, mu=0.01):
    """
    Train using FedProx (Federated Proximal)
    FedProx adds a proximal term to client optimization (μ parameter)
    """
    print(f"Starting FedProx training with μ={mu}")
    
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
            
            # Train one client (Note: FedProx proximal term would normally be added to the client optimization
            # but since we can't modify Client.py, we're implementing the FedProx concept here)
            data_size = client.train_epoch(cid=client_id)
            
            # Get client model variables
            client_vars = client.get_client_vars()
            
            # Apply proximal term effect (simplified implementation since we can't modify the optimization)
            # In true FedProx, the proximal term would be added to the loss function
            proximal_client_vars = []
            for cv, gv in zip(client_vars, global_vars):
                # Apply proximal regularization effect by pulling weights toward global model
                proximal_cv = cv - mu * (cv - gv)
                proximal_client_vars.append(proximal_cv)
            
            # Sum up weighted client vars
            weight = data_size / total_data_size
            
            if client_vars_sum is None:
                client_vars_sum = [weight * x for x in proximal_client_vars]
            else:
                for cv_sum, pcv in zip(client_vars_sum, proximal_client_vars):
                    cv_sum += weight * pcv
        
        # Update global vars
        global_vars = client_vars_sum
        
        # Test and record results
        acc, loss = run_global_test(client, global_vars, test_num, ep)
        results['accuracy'].append(acc)
        results['loss'].append(loss)
    
    # Final evaluation
    final_acc, final_loss = run_global_test(client, global_vars, 10000, epochs-1)
    print(f"Final FedProx (μ={mu}) results - Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
    
    return results, final_acc, final_loss 