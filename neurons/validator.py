# The MIT License (MIT)
# Copyright © 2023 Daniel Leon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Bittensor Validator Template:

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import traceback
import bittensor as bt
import pickle

from map_reduce.protocol import MapReduce
from neurons.gradient_compute import average_gradient, update_scores
from neurons.dataset import dataloader, reduce_map
from neurons.model import *
from neurons.preprocess import prep_data

# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
def get_config():

    parser = argparse.ArgumentParser()
    # Adds override arguments for network and netuid.
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    # Adds arguments for model name.
    # parser.add_argument( '--model_name', type = str, help = "Huggingface pretrained model name to train." )
    # Adds arguments for dataset name.
    parser.add_argument( '--dataset', type = str, help = "Huggingface dataset to train the model on." )
    # Adds arguments for name of loss function.
    parser.add_argument( '--loss_fn', type = str, help = "Loss function to use during training." )
    # Adds arguments for batch size.
    parser.add_argument( '--batch_size', type = int, default = 32, help = "The chain subnet uid." )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/miner.py --help
    config =  bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'validator',
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config

def main( config ):
    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet( config = config )
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor( config = config )
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite( wallet = wallet )
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other miners.
    metagraph = subtensor.metagraph( config.netuid )
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again.")
        exit()
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")
        # ADD : set validators permit to True
        metagraph.validator_permit[my_subnet_uid] = True

    # Step 6: Set up initial scoring weights for validation
    bt.logging.info("Building validation weights.")
    alpha = 0.9

    # Define two scores; reliance_scores and capacity_scores
    # The reliance_scores represents how correctly each miner is computing
    # The capacity_scores represents how much data each miner can handle at the same time
    # The reliance_score of validators and other  who return None or malicious response will gradually converge to 0
    reli_scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    capacity_scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    capacity_scores = capacity_scores / torch.sum(capacity_scores)

    bt.logging.info(f"Reliance weights of miners: {reli_scores}")

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0

    # Define the interval of model updates for miners
    update_step = 2

    # Define the training dataset dataloader
    try:
        train_dataloader = dataloader(config.dataset, config.batch_size, True)
    except:
        bt.logging.info("Unable to load the dataset. Choose another one.")
        return

    # Define the dataset preprocessor
    preprocessor = build_processor()

    # # Define the resnet18 model and optimizer.
    model = build_model()
    optimizer = build_optimizer(model, lr=0.1)

    # Dump the model to the local storage
    model_dump_path = '/home/ubuntu/subnet-template/model.pkl'
    with open(model_dump_path, 'wb') as f:
        pickle.dump(model, f)

    while True:
        try:
            # Broadcast a query to all miners on the network
            # Set the model_update_flag to True at every update_step
            if step % update_step == 0:
                model_update_flag = True
                with open(model_dump_path, 'wb') as f:
                    pickle.dump(model, f)
            else:
                model_update_flag = False

            # Preprocess input data
            data = next(iter(train_dataloader))
            imgs, labels = prep_data(data, preprocessor)

            # Segment the data according to capacity_scores
            if len(capacity_scores) == 0:
                bt.logging.info(f"No valid miner available")
                continue
            data_segs = reduce_map(labels, capacity_scores)

            # Send requests to synapses
            # The entire batch and corresponding segments are parsed to send requests simultaneously.
            reli_scores_list = list(bt.Tensor.serialize(reli_scores).numpy()) # ISSUE
            responses = dendrite.query(
                metagraph.axons,
                MapReduce(
                    input_data = [bt.Tensor.serialize(imgs),
                                   bt.Tensor.serialize(labels)],
                    reli_scores = reli_scores_list,
                    input_segs = data_segs,
                    loss_fn = config.loss_fn,
                    update_model = model_update_flag,
                    model_path = model_dump_path),
                    deserialize = True
            )

            # Calculate the average gradient using responses from miners
            avg_grad = average_gradient(responses, reli_scores)

            # Update the model parameter using average gradient
            if avg_grad != None:
                for j, param in enumerate(model.parameters()):
                    param.grad = avg_grad[j]
                optimizer.step()

                # Update the reliance score
                reli_scores = update_scores(responses, avg_grad, reli_scores)
                bt.logging.info(f"Reliance weights of miners: {reli_scores}")

            # Periodically update the weights on the Bittensor blockchain.
            if (step + 1) % 20 == 0:
                weights = torch.nn.functional.normalize(reli_scores, p=1.0, dim=0)
                bt.logging.info(f"Setting weights: {weights}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = subtensor.set_weights(
                    netuid = config.netuid, # Subnet to set weights on.
                    wallet = wallet, # Wallet to sign set weights using hotkey.
                    uids = metagraph.uids, # Uids of the miners to set weights for.
                    weights = weights, # Weights to set for the miners.
                    wait_for_inclusion = True # origin : True
                )
                if result: bt.logging.success('Successfully set weights.')
                else: bt.logging.error('Failed to set weights.')

            # End the current step and prepare for the next iteration.
            step += 1
            # Resync our local state with the latest state from the blockchain.
            metagraph = subtensor.metagraph(config.netuid)
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            # time.sleep(bt.__blocktime__)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config()
    # Run the main function.
    main( config )
