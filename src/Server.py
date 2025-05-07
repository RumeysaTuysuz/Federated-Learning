import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm

from Client import Clients

# Only define functions needed by other modules

def buildClients(num):
    """Builds the Clients object with specified parameters."""
    learning_rate = 0.0001
    num_input = 32
    num_input_channel = 3
    num_classes = 10

    return Clients(input_shape=[None, num_input, num_input, num_input_channel],
                   num_classes=num_classes,
                   learning_rate=learning_rate,
                   clients_num=num)


def run_global_test(client, global_vars, test_num):
    """Sets global vars for a client and runs a test."""
    # Need to handle the 'ep' variable issue if this function is still used elsewhere
    # For now, removing the reference to 'ep' which caused errors before
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[{} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        test_num, acc, loss))
    return acc, loss

# --- Removed Training Execution Code --- #
# The following code was removed as it runs training directly when imported.
# Training logic is now handled by compare_methods.py or specific training scripts.
# 
# CLIENT_NUMBER = 100
# CLIENT_RATIO_PER_ROUND = 0.12
# epoch = 360
# 
# client = buildClients(CLIENT_NUMBER)
# 
# global_vars = client.get_client_vars()
# for ep in range(epoch):
#     client_vars_sum = None
#     # ... (rest of the training loop) ...
# 
# run_global_test(client, global_vars, test_num=10000)
# --- End of Removed Code --- #