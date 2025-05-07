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

def run_global_test(client, global_vars, test_num):
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[{} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        test_num, acc, loss))
    return acc, loss

def train_fedprox(client_number, client_ratio_per_round, epochs, test_num=600, mu=0.01, batch_size=32):
    """
    Train using FedProx (Federated Proximal)
    FedProx adds a proximal term to client optimization (μ parameter)
    """
    print(f"Starting FedProx training with μ={mu}, batch_size={batch_size}")

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
        
        for client_id in tqdm(random_clients, ascii=True, desc=f"Epoch {ep+1}/{epochs}"):
            client.set_global_vars(global_vars)
            
            data_size = client.train_epoch(cid=client_id, batch_size=batch_size)

            client_vars = client.get_client_vars()
      
            proximal_client_vars = []
            for cv, gv in zip(client_vars, global_vars):
                proximal_cv = cv - mu * (cv - gv)
                proximal_client_vars.append(proximal_cv)

            weight = data_size / total_data_size
            
            if client_vars_sum is None:
                client_vars_sum = [weight * x for x in proximal_client_vars]
            else:
                for cv_sum, pcv in zip(client_vars_sum, proximal_client_vars):
                    cv_sum += weight * pcv

        global_vars = client_vars_sum

        acc, loss = run_global_test(client, global_vars, test_num)
        results['accuracy'].append(acc)
        results['loss'].append(loss)

    final_acc, final_loss = run_global_test(client, global_vars, 10000)
    print(f"Final FedProx (μ={mu}) results - Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
    
    return results, final_acc, final_loss 