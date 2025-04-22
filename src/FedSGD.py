import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
import numpy as np

from Client import Clients

def buildClients(num):
    learning_rate = 0.0001
    num_input = 32 
    num_input_channel = 3  
    num_classes = 10 

 
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

    client = buildClients(client_number)
    
    results = {
        'accuracy': [],
        'loss': []
    }
    
    global_vars = client.get_client_vars()
    
    for ep in range(epochs):
        random_clients = client.choose_clients(client_ratio_per_round)
        total_data_size = sum([client.dataset.train[cid].size for cid in random_clients])

        client_vars_sum = None
        
        for client_id in tqdm(random_clients, ascii=True):
            client.set_global_vars(global_vars)
            
            batch_x, batch_y = client.dataset.train[client_id].next_batch(batch_size)

            data_size = client.train_epoch(cid=client_id)
     
            client_vars = client.get_client_vars()
            
            weight = data_size / total_data_size
            
            if client_vars_sum is None:
                client_vars_sum = [weight * x for x in client_vars]
            else:
                for cv_sum, cv in zip(client_vars_sum, client_vars):
                    cv_sum += weight * cv
        
        global_vars = client_vars_sum
        
        acc, loss = run_global_test(client, global_vars, test_num, ep)
        results['accuracy'].append(acc)
        results['loss'].append(loss)
    
    final_acc, final_loss = run_global_test(client, global_vars, 10000, epochs-1)
    print(f"Final FedSGD results - Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
    
    return results, final_acc, final_loss 