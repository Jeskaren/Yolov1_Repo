# -*- coding:utf-8 -*-
# 一万年太久，只争朝夕
# Jeskaren
from torch.utils.tensorboard import SummaryWriter

from yolo_Data import yoloDataset
from yolo_Loss import yoloLoss
from DarkNet import darknet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

device = 'cuda'
file_root = 'VOCdevkit/VOC2007/JPEGImages/'
batch_size = 50
momentum = 0.9
weight_decay = 0.0005

# 自定义训练数据集
train_dataset = yoloDataset(img_root=file_root, list_file='Data_Set/Voc_train.txt', train=True, transform=[transforms.ToTensor()])
# 加载自定义的训练数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# 自定义测试数据集
test_dataset = yoloDataset(img_root=file_root, list_file='Data_Set/Voc_test.txt', train=False, transform=[transforms.ToTensor()])
# 加载自定义的测试数据集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print('the dataset has %d images' % (len(train_dataset)))


net = darknet()
net = net.to(device)

criterion = yoloLoss(7, 2, 5, 0.5)
criterion = criterion.to(device)
net.train()


def adjust_learning_rate(optimizer, epoch):
    if epoch < 1:
        lr = 1e-3 + (1e-2 - 1e-3) * epoch  # 从1e-3到1e-2线性增长
    elif epoch < 76:
        lr = 1e-2
    elif epoch < 106:
        lr = 1e-3
    else:
        lr = 1e-4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 定义优化器
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=1e-3,  # 初始学习率
    momentum=momentum,
    weight_decay=weight_decay
)


writer = SummaryWriter(log_dir='./logs_darknet')
num_epochs = 10
for epoch in range(num_epochs):
    net.train()
    lr = adjust_learning_rate(optimizer, epoch)
    print(f'\n\nStarting epoch {epoch + 1} / {num_epochs}')
    print(f'Learning Rate for this epoch: {lr}')

    total_loss = 0.
    step = 0
    for i, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)
        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.item()
        step += 1
        writer.add_images('images', images, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Iter [{i + 1}/{len(train_loader)}] Loss: {loss.item():.4f}, average_loss: {total_loss / (i + 1):.4f}')
            writer.add_scalar('Loss/train', loss.item(), step)


    step = 0
    net.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images, target = images.to(device), target.to(device)
            pred = net(images)
            loss = criterion(pred, target)
            validation_loss += loss.item()
    validation_loss /= len(test_loader)

    print(f'Validation loss after epoch {epoch + 1}: {validation_loss:.5f}')

    # 保存模型参数
    # torch.save(net.state_dict(), 'yolo.pth')
    torch.save(net.state_dict(), 'yolo_darknet_1.pth')
