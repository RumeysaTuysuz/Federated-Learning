import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tqdm import tqdm

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
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))


CLIENT_NUMBER = 100
CLIENT_RATIO_PER_ROUND = 0.12
epoch = 360

client = buildClients(CLIENT_NUMBER)

global_vars = client.get_client_vars()
for ep in range(epoch):
    client_vars_sum = None

    random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
    total_data_size = sum([client.dataset.train[cid].size
                            for cid in random_clients])

    for client_id in tqdm(random_clients, ascii=True):
        client.set_global_vars(global_vars)

        data_size = client.train_epoch(cid=client_id)

        current_client_vars = client.get_client_vars()

        if client_vars_sum is None:
            weight = data_size / total_data_size
            client_vars_sum = [weight * x for x in current_client_vars]
        else:
            for cv, ccv in zip(client_vars_sum, current_client_vars):
                cv += weight * ccv

    global_vars = client_vars_sum

    run_global_test(client, global_vars, test_num=600)


run_global_test(client, global_vars, test_num=10000)