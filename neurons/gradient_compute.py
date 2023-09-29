import torch
import typing
import bittensor as bt

# Compute the weighted average gradient of gradients from several miners
def average_gradient(gradients: typing.Optional[typing.List[bt.Tensor]], weights: torch.FloatTensor):
    # Each miner returns a gradient that is a list of gradients from each layer of the model

    # Check valid gradients and convert gradients from bittensor Tensor to torch Tensor
    valid_gradients = []
    valid_weights = []
    for i, gradient in enumerate(gradients):
        if gradient!=None:
            for j, grad in enumerate(gradient):
                gradient[j] = grad.deserialize()
            valid_gradients.append(gradient)
            valid_weights.append(weights[i])

    # Calculate the weighted average of the gradients using reliance scores
    if len(valid_gradients) > 0:
        grad_center = weighted_average(valid_gradients, valid_weights)
        return grad_center
    else:
        return None

# Compute the weighted average using weights
def weighted_average(gradients: typing.Optional[typing.List[torch.FloatTensor]], weights: typing.List[torch.FloatTensor]):
    grad_center = []
    total_score = 1e-10
    grad_dim = len(gradients[0])
    for i in range(grad_dim):
        grad_center.append(torch.Tensor())
    for i, gradient in enumerate(gradients):
        for j, grad in enumerate(gradient):
            grad_center[j] = torch.cat((grad_center[j], grad.unsqueeze(0)*weights[i]), 0)
        total_score += weights[i]

    for j in range(grad_dim):
        grad_center[j] = torch.sum(grad_center[j], dim=0) / total_score

    return grad_center

# Update scores regarding the distance from the average gradient to each gradient
def update_scores(gradients, avg_grad, scores):
    alpha = 0.9
    for i, gradient in enumerate(gradients):
        if gradient==None:
            score = 0
        else:
            dist = relative_dist(gradient, avg_grad)
            score = torch.max(torch.tensor(0.3), 1 - dist)

        scores[i] = alpha*scores[i] + (1 - alpha) * score

    return scores

# Calculate relative distance from the average gradient
def relative_dist(gradient, avg_grad):
    dist = torch.tensor(1e-10)
    for i, grad in enumerate(gradient):
        dist = torch.max(dist, torch.norm(grad - avg_grad[i]) / torch.norm(avg_grad[i]))

    return dist

# All available loss functions
__all__ = ['L1Loss', 'NLLLoss', 'NLLLoss2d', 'PoissonNLLLoss', 'GaussianNLLLoss', 'KLDivLoss',
           'MSELoss', 'BCELoss', 'BCEWithLogitsLoss', 'HingeEmbeddingLoss', 'MultiLabelMarginLoss',
           'SmoothL1Loss', 'HuberLoss', 'SoftMarginLoss', 'CrossEntropyLoss', 'MultiLabelSoftMarginLoss',
           'CosineEmbeddingLoss', 'MarginRankingLoss', 'MultiMarginLoss', 'TripletMarginLoss',
           'TripletMarginWithDistanceLoss', 'CTCLoss']

# Feed the input to the model and compute the loss and gradient
# Used in miners
def compute_grads(model, input, target, loss_fn):
    try:
        softmax = torch.nn.Softmax(dim=1)
        output = softmax(model(input)['logits'])

        # Check if loss function is valid function
        if not loss_fn in __all__:
            bt.logging.info("Loss function unavailable. Use another one.")
            return 0, None

        # Compute the loss and gradient
        loss_fn_ = getattr(torch.nn, loss_fn)()
        loss = loss_fn_(output, target)
        loss.backward()
        grads = [param.grad for param in model.parameters()]

        return loss, grads
    except:
        bt.logging.info("Error while computing loss and gradients")
        return 0, None
