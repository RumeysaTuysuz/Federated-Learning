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

def train_fednova(client_number, client_ratio_per_round, epochs, test_num=600, local_updates=20, batch_size=32):
    """
    Train using FedNova (Normalized Averaging)
    FedNova addresses the objective inconsistency in FedAvg by normalizing updates
    """
    print(f"Starting FedNova training with batch_size={batch_size}")

    client = buildClients(client_number)

    results = {
        'accuracy': [],
        'loss': []
    }
    
    global_vars = client.get_client_vars()
    
    for ep in range(epochs):
        random_clients = client.choose_clients(client_ratio_per_round)
        
        client_weights = []  
        client_vars_deltas = []  
        client_tau_effs = []  

        for client_id in tqdm(random_clients, ascii=True, desc=f"Epoch {ep+1}/{epochs}"):
            client.set_global_vars(global_vars)

            init_vars = client.get_client_vars()

            data_size = client.train_epoch(cid=client_id, batch_size=batch_size)
            client_weights.append(data_size)

            updated_vars = client.get_client_vars()
        
            model_delta = [updated - init for updated, init in zip(updated_vars, init_vars)]
            client_vars_deltas.append(model_delta)

            # For FedNova, we need to estimate the effective number of local steps
            # In practice, this would be calculated based on the actual number of gradient steps
            # Since we can't modify Client.py, we'll use a simple heuristic based on data_size
            # Assuming batch_size used in train_epoch is the one passed to this function
            tau_eff = data_size / batch_size # Estimate based on batch size used
            client_tau_effs.append(tau_eff)

        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        total_tau_eff = sum(client_tau_effs)
        avg_tau_eff = total_tau_eff / len(client_tau_effs)
        nova_coeffs = [normalized_weights[i] * (client_tau_effs[i] / avg_tau_eff) 
                     for i in range(len(normalized_weights))]

        for var_idx in range(len(global_vars)):
            weighted_delta = sum(nova_coeffs[i] * client_vars_deltas[i][var_idx] for i in range(len(nova_coeffs)))
            global_vars[var_idx] = np.add(global_vars[var_idx], weighted_delta, casting="unsafe")

        acc, loss = run_global_test(client, global_vars, test_num)
        results['accuracy'].append(acc)
        results['loss'].append(loss)

    final_acc, final_loss = run_global_test(client, global_vars, 10000)
    print(f"Final FedNova results - Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
    
    return results, final_acc, final_loss 