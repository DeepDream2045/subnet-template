import torch
import datasets

# Define dataloader based on the database name
def dataloader(dataset_name='cifar10', batch_size=64, shuffle=True):
    kwargs = {'batch_size': batch_size, 'shuffle': shuffle}
    dataset = datasets.load_dataset(dataset_name, split='train').with_format('torch')
    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    return dataloader

# Define reduce_map function that allocates the whole dataset to each node
def reduce_map(data, scores):
    data_segs = [0]
    data_length = data.shape[0]
    total_score = sum(scores)
    for score in scores:
        data_segs.append(min(int(data_length * score / total_score) + data_segs[-1], data_length))

    return data_segs