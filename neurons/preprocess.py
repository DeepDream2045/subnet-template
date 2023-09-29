# Preprocess the data using preprocessor
def prep_data(batch, preprocessor):
    imgs, labels = prep_cifar10(batch, preprocessor)
    return imgs, labels

# The cifar10 preprocessing function
def prep_cifar10(batch, cifar10_processor):
    raw_imgs = batch['img']
    labels = batch['label']
    imgs = cifar10_processor(raw_imgs, return_tensors='pt')['pixel_values']

    return imgs, labels