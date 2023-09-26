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

import typing
import bittensor as bt

# This is the protocol for the miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a response.

class MapReduce(bt.Synapse):
    """
    A simple MapReduce protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    - mr_input: An integer value representing the input request sent by the validator.
    - reliance_score : A list of reliance scores of all the miners the validator is sending requests.
    - input_segs : A list of segments of input data according to miners the validator is sending requests.
    - update_model : A flag to show if the model should be updated.
    - model_path : A path of the pickle model dump file.
    - mr_output: An optional integer value which, when filled, represents the response from the miner.
    """

    # Required request input, filled by sending dendrite caller.
    mr_input: typing.List[bt.Tensor]

    # Reward score assessed by the validator, filled by sending dendrite caller.
    reliance_score: typing.List[float]

    # Batch segmentation
    input_segs: list

    # Model update flag. If True, each miner should reload the model updated by the validator first.
    update_model: bool

    # Model path for the miner. Validator stores the model to this path and let the miner know it.
    model_path: str

    # Optional request output, filled by recieving axon.
    mr_output: typing.Optional[typing.List[bt.Tensor]] = None

    class Config:
        arbitrary_types_allowed=True

    def deserialize(self) -> bt.Synapse:
        """
        Deserialize the MapReduce output. This method retrieves the response from
        the miner in the form of mr_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - bittensor.Tensor: The deserialized response, which in this case is the value of mr_output.

        Example:
        Assuming a miner instance has a mr_output value of 5:
        >>> mr_instance = MapReduce(mr_input=4)
        >>> mr_instance.mr_output = 5
        >>> mr_instance.deserialize()
        5
        """
        return self.mr_output
