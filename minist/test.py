from torchvision import transforms,utils
from torchvision import datasets
from torch.utils import data
import matplotlib.pyplot as plt

# my_trans=transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])
my_trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_data = datasets.ImageFolder('D:/Pycharm/torchvision_data/torchvision_data', transform=my_trans)
train_loader = data.DataLoader(train_data, batch_size=8, shuffle=True, )


for i_batch, img in enumerate(train_loader):
    print(i_batch)
    print(img)
    # if i_batch == 0:
    #     print(img[1])
    #     fig = plt.figure()
    #     grid = utils.make_grid(img[0])
    #     plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #     plt.show()
    #     utils.save_image(grid, 'test01.png')
    # break

# from torchvision import transforms, utils
# from torchvision import datasets
# import torch
# import matplotlib.pyplot as plt
# from torch.utils import data
#
# my_trans = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])
# train_data = datasets.ImageFolder('D:/Pycharm/torchvision_data/torchvision_data', transform=my_trans)
# train_loader = data.DataLoader(train_data, batch_size=8, shuffle=True, )
#
# for i_batch, img in enumerate(train_loader):
#     if i_batch == 0:
#         print(img[1])
#         fig = plt.figure()
#         grid = utils.make_grid(img[0])
#         plt.imshow(grid.numpy().transpose((1, 2, 0)))
#         plt.show()
#         utils.save_image(grid, 'test01.png')
#     break
