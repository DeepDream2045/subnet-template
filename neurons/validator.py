# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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
# TODO(developer): Rewrite based on protocol defintion.

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import traceback
import bittensor as bt

# import this repo
import template
# ADD : Import necessary libraries
# from torchvision import datasets, transforms
import datasets as datasets
import wandb

# END ADD

# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
def get_config():

    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
    # Adds override arguments for network and netuid.
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
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

    # ADD
    # Define two scores; reliance_scores and capacity_scores
    # reliance_scores shows how correctly each miner is computing.
    # capacity_scores shows how much data each miner can handle at the same time.
    # The reliance_score of validators and other miners who return None will gradually become 0.
    reliance_scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    capacity_scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    capacity_scores = capacity_scores / torch.sum(capacity_scores)
    # End ADD

    bt.logging.info(f"Reliance weights of miners: {reliance_scores}")

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0

    # TODO ADD
    # Define the interval of model updates for miners
    update_step = 2

    # Define the cifar-10 dataloader for training dataset
    train_kwargs = {'batch_size': len(metagraph.S) * 64, 'shuffle': True}
    cifar_dataset = datasets.load_dataset("cifar10", split='train[:640]').with_format('torch')
    cifar_dataloader = torch.utils.data.DataLoader(cifar_dataset, **train_kwargs)

    # Define the resnet18 model and model checkpoint path.
    from transformers import ResNetModel, ResNetConfig, ResNetForImageClassification, ConvNextFeatureExtractor
    resnet_config = ResNetConfig(
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        layer_type='basic',
        hidden_act='relu',
        downsample_in_first_stage=False,
        out_features=['stage4'],
        out_indices=[4],
        num_labels=10,
        label2id={'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9},
        id2label={0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    )

    img_processor = ConvNextFeatureExtractor(size=32,
                                             image_mean=[0.485, 0.456, 0.406],
                                             image_std=[0.229, 0.224, 0.225])


    # # Define the model and model checkpoint path.
    model = ResNetForImageClassification(resnet_config)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

    model_ckpt_path = '/home/ubuntu/subnet-template/model.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_ckpt_path)
    # END ADD

    while True:
        try:
            # TODO(developer): Define how the validator selects a miner to query, how often, etc.
            # Broadcast a query to all miners on the network.
            # SHOULD separate the input according to capacity_scores for each axon.
            # Check if miner model should be updated.
            if step % update_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_ckpt_path)

            # Set input to each axon and get responses each
            data = next(iter(cifar_dataloader))
            images = data['img']
            # labels = torch.nn.functional.one_hot(data['label'], num_classes=10)
            labels = data['label']
            images = img_processor(images, return_tensors='pt')['pixel_values']

            data_len = images.shape[0]
            data_segs = [0]
            total_capa_score = sum(capacity_scores)
            for capa_score in capacity_scores:
                data_segs.append(min(int(data_len * capa_score / total_capa_score) + data_segs[-1], data_len))

            responses = dendrite.query(
                metagraph.axons,
                template.protocol.Dummy(
                    dummy_input = [bt.Tensor.serialize(images),
                                   bt.Tensor.serialize(labels)],
                    dummy_score = 1.0,
                    dummy_segs = data_segs,
                    dummy_update = True,
                    dummy_model_path = model_ckpt_path),
                deserialize = True

            )
            for i, response in enumerate(responses):
                if response!=None:
                    for j, grad in enumerate(response):
                        response[j] = grad.deserialize()
                    responses[i]=response

            bt.logging.info(f"Received dummy responses")

            # # TODO(developer): Define how the validator scores responses.
            # TODO : ADD
            # Calculate the center of the result using reliance scores.
            center = []
            total_score = 1e-10
            grad_dim = 0
            # Get the length of the grads
            for i, resp_i in enumerate(responses):
                if resp_i != None:
                    grad_dim = len(resp_i)
                    break

            # Calculate the center of gradients in case at least one
            if grad_dim > 0:
                # Make the list of grads
                grads_list = []
                valid_grads_ids = []
                for i, resp_i in enumerate(responses):
                    if resp_i != None:
                        valid_grads_ids.append(i)
                        grads_list.append(resp_i)

                # Calculate the center of the gradients
                for i in range(grad_dim): center.append(torch.Tensor())
                for i, grads in enumerate(grads_list):
                    for j, grad in enumerate(grads):
                        center[j] = torch.cat((center[j], grad.unsqueeze(0) * reliance_scores[i]), 0)
                    total_score += reliance_scores[i]

                for j in range(grad_dim):
                    center[j] = torch.sum(center[j], dim=0) / total_score

                # Apply backpropagation using gradients center
                for j, param in enumerate(model.parameters()):
                    param.grad = center[j]
                optimizer.step()

                # Calculate the distance between each response and the center
                distances = []
                for i, grads in enumerate(grads_list):
                    distances.append([])
                    for j in range(grad_dim):
                        distances[-1].append(torch.max(torch.norm(center[j] - grads[j]), torch.tensor(1e-10)))

                # Normalize the distances
                distances = torch.tensor(distances)
                for i in range(grad_dim):
                    distances[:, i] = distances[:, i] / torch.max(distances[:, i])

                # Calculate the score of each response according to the distance to the center.
                for i, resp_i in enumerate(responses):
                    if resp_i==None:
                        score=0
                    else:
                        dist = torch.max(distances[:, valid_grads_ids.index(i)])

                        score = 1 - dist / 2
                    reliance_scores[i] = alpha * reliance_scores[i] + (1 - alpha) * score

            # Normalize the reliance_scores to avoid the shrink of reliance_scores.
            reliance_scores = reliance_scores / torch.max(reliance_scores)
            capacity_scores =  torch.mul(capacity_scores, reliance_scores)
            capacity_scores = capacity_scores / torch.sum(capacity_scores)
            bt.logging.info(f"Reliance scores: {reliance_scores}")
            # END ADD

            # Periodically update the weights on the Bittensor blockchain.
            if (step + 1) % 2000 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.nn.functional.normalize(reliance_scores, p=1.0, dim=0)
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

    model_wandb.finish()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config()
    # Run the main function.
    main( config )
