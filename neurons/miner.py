# The MIT License (MIT)
# Copyright © 2023 Daniel Leon

# This license grants permission, free of charge, to any person obtaining a copy of this software and associated
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

# Bittensor Miner Template:
# TODO(developer): Rewrite based on protocol and validator definition.

# Step 1: Import necessary libraries and modules
import os
import time
import argparse
import traceback
import bittensor as bt
import typing
import torch
import pickle
import map_reduce
from neurons.gradient_compute import compute_grads

def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # Adds override arguments for network and netuid.
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'miner',
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)
    return config


# Main takes the config and starts the miner.
def main( config ):

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet( config = config )
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor( config = config )
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again. ")
        exit()
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Step 4: Set up miner functionalities
    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def blacklist_fn(synapse: map_reduce.protocol.MapReduce) -> typing.Tuple[bool, str]:
        """
        This function defines how miners should blacklist requests. It runs before the synapse data has been
        deserialized (i.e., before synapse.data is available). The synapse is constructed via the headers of the request.
        It is important to blacklist requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (map_reduce.protocol.MapReduce): The synapse object containing the request data.

        Returns:
            typing.Tuple[bool, str]: A tuple containing a boolean indicating whether to blacklist the request and a string message explaining the decision.
        """

        # Check if the hotkey is a registered entity in the metagraph
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return(True, f"Unrecognized hotkey {synapse.dendrite.hotkey}")

        # Check if the validator has a permission to validate
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        if not metagraph.validator_permit[caller_uid]:
            bt.logging.trace(f'Blacklisting invalid validator {synapse.dendrite.hotkey}')
            return(True, f"No validation permission {synapse.dendrite.hotkey}")

        # Check if the validator has enough stakes
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        stake = float( metagraph.S[ caller_uid ] * synapse.reli_scores[caller_uid]) # Return the stake as the priority.
        if stake == 0.0:
            # Ignore requests from validators with empty wallet
            bt.logging.trace(f'Blacklisting empty wallet {synapse.dendrite.hotkey}')
            return(True, f"Empty wallet validator {synapse.dendrite.hotkey}")

        # TODO: Add additional checks here if necessary

        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return(False, f"Valid validator {synapse.dendrite.hotkey}")

    def priority_fn(synapse: map_reduce.protocol.MapReduce) -> float:
        """
        This function defines how miners should prioritize requests. It runs after the blacklist function has been called.
        Miners may receive messages from multiple entities at once. This function determines which request should be processed first.
        Higher values indicate that the request should be processed first. Lower values indicate that the request should be processed later.
        
        Args:
            synapse (map_reduce.protocol.MapReduce): The synapse object containing the request data.
        
        Returns:
            float: The priority score of the request. Higher values indicate higher priority.
        """
        # Get the caller index.
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        # Calculate the priority score based on the stake and reliance score of the caller.
        # Higher stake and reliance score leads to higher priority.
        priority = float(metagraph.S[caller_uid] * synapse.reli_scores[caller_uid])
        
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', priority)
        
        return priority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def compute_gradient(synapse: map_reduce.protocol.MapReduce) -> map_reduce.protocol.MapReduce:
        """
        This function defines how miners should process requests. It runs after the synapse has been deserialized 
        (i.e., after synapse.data is available) and after the blacklist and priority functions have been called.
        
        Args:
            synapse (map_reduce.protocol.MapReduce): The synapse object containing the request data.
        
        Returns:
            map_reduce.protocol.MapReduce: The synapse object with the calculated gradients.
        """


        # Check if miner should update the model and update it.
        if synapse.update_model:
            with open(synapse.model_path, 'rb') as f:
                model = pickle.load(f)

        # Check if model exists
        try:
            model
        except NameError:
            try:
                with open(synapse.model_path, 'rb') as f:
                    model = pickle.load(f)
            except:
                synapse.gradients = None
                bt.logging.info("Model does not exist or is corrupted")
                return synapse

        # Get map reduced data using input_segs
        try:
            input, target = synapse.input_data
            input = bt.Tensor.deserialize(input)
            input = input[synapse.input_segs[my_subnet_uid]:synapse.input_segs[my_subnet_uid+1]]
            target = bt.Tensor.deserialize(target)
            target = target[synapse.input_segs[my_subnet_uid]:synapse.input_segs[my_subnet_uid + 1]]
        except:
            bt.logging.info(f"Incorrect input data type")
            synapse.gradients = None
            return synapse

        # Calculate the loss and gradients
        loss, grads = compute_grads(model, input, target, synapse.loss_fn)
        if grads == None:
            synapse.gradients = None
            return synapse

        bt.logging.info(f"Loss : {loss}")

        # Return gradients
        synapse.gradients = []
        for grad in grads:
            synapse.gradients.append(bt.Tensor.serialize(grad))

        return synapse


    # Step 5: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    # Added the port configuration to use multiple miners.
    axon = bt.axon( wallet = wallet, port = config.axon.port ) # origin : axon = bt.axon( wallet = wallet )
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn = compute_gradient,
        blacklist_fn = blacklist_fn,
        priority_fn = priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving axon {map_reduce} on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve( netuid = config.netuid, subtensor = subtensor )

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 6: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log =  (f'Step:{step} | '\
                        f'Block:{metagraph.block.item()} | '\
                        f'Stake:{metagraph.S[my_subnet_uid]} | '\
                        f'Rank:{metagraph.R[my_subnet_uid]} | '\
                        f'Trust:{metagraph.T[my_subnet_uid]} | '\
                        f'Consensus:{metagraph.C[my_subnet_uid] } | '\
                        f'Incentive:{metagraph.I[my_subnet_uid]} | '\
                        f'Emission:{metagraph.E[my_subnet_uid]}')
                bt.logging.info(log)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success('Miner killed by keyboard interrupt.')
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main( get_config() )
