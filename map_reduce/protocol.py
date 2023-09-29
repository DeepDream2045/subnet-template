"""
MIT License (MIT)
Copyright © 2023 Daniel Leon

This software is provided 'as is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
"""

import typing
import bittensor as bt

"""
This module defines the protocol for the miner and validator.
It is a simple request-response protocol where the validator sends a request
to the miner, and the miner responds with a response.
"""

class MapReduce( bt.Synapse ):
    input_data: typing.List[bt.Tensor]  # Required request input data, filled by sending dendrite caller.

    reli_scores: typing.List[float]  # Reward score assessed by the validator, filled by sending dendrite caller.

    input_segs: list  # Batch segmentation

    update_model: bool  # Model update flag. If True, each miner should reload the model updated by the validator first.

    model_path: str  # Model path for the miner. Validator stores the model to this path and let the miner know it.

    loss_fn: str  # Model path for the miner. Validator stores the model to this path and let the miner know it.

    gradients: typing.Optional[typing.List[bt.Tensor]] = None  # Optional request output, filled by receiving axon.

    def deserialize(self) -> bt.Synapse:
        """
        Deserialize responses.
        Returns:
            bt.Synapse: The deserialized response.
        """
        # TODO: Add error handling for deserialization
        return self.gradients
