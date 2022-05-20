import matplotlib.pyplot as plt
import numpy as np
import torchvision
def show_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def validate_dataloader(dataloader):
    image, metadata, target = next(iter(dataloader))
    for (img, metadata, target) in dataloader:
        print(img.shape, metadata.shape, target.shape)
        break
    show_image(torchvision.utils.make_grid(image))
