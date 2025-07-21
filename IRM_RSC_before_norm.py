#这个程序集成了RSC/IRM/baseline，和最终版本的不同是架构没有更改，就是最基本的LeNet5，
#大家的准确率都是70左右，并且baseline更好
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
import random
from torch.utils.data import Subset,DataLoader,ConcatDataset
from torchvision.datasets import FakeData
from itertools import cycle
import torch.nn.functional as F

def color_grayscale_arr(arr, forground_color, background_color):
    """Converts grayscale image"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])#增加一个“通道”维度
    if background_color == "black":
        if forground_color == "red":
            arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)#创建全零数组作为绿色和蓝色通道，表示全红色
        elif forground_color == "green":
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif forground_color == "white":
            arr = np.concatenate([arr, arr, arr], axis=2)
    else:
        if forground_color == "yellow":
            arr = np.concatenate([arr, arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
        else:
            arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype), arr], axis=2)

        c = [255, 255, 255]
        arr[:, :, 0] = (255 - arr[:, :, 0]) / 255 * c[0]
        arr[:, :, 1] = (255 - arr[:, :, 1]) / 255 * c[1]
        arr[:, :, 2] = (255 - arr[:, :, 2]) / 255 * c[2]

    return arr


class ColoredMNIST(datasets.VisionDataset):

    def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.prepare_colored_mnist()
        if env in ['train1', 'train2', 'train3', 'test1', 'test2']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt',
                                               weights_only=False)
        elif env == 'all_train':
            train1_data = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt'),
                                                weights_only=False ) 
            train2_data=torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'),
                                                weights_only=False)
            train3_data=torch.load(os.path.join(self.root, 'ColoredMNIST', 'train3.pt'),
                                                weights_only=False)
            self.data_label_tuples = train1_data + train2_data + train3_data
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, train3, test1, test2, and all_train')

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'train3.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'test1.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'test2.pt')):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        train2_set = []
        train3_set = []
        test1_set, test2_set = [], []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)
            
            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Color the image according to its environment label

            if idx < 10000:
                colored_arr = color_grayscale_arr(im_array, forground_color = "red", background_color = "black")
                train1_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 20000:
                colored_arr = color_grayscale_arr(im_array, forground_color = "green", background_color = "black")
                train2_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 30000:
                colored_arr = color_grayscale_arr(im_array, forground_color = "white", background_color = "black")
                train3_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 45000:
                colored_arr = color_grayscale_arr(im_array, forground_color = "yellow", background_color = "white")
                test1_set.append((Image.fromarray(colored_arr), binary_label))
            else:
                colored_arr = color_grayscale_arr(im_array, forground_color = "blue", background_color = "white")
                test2_set.append((Image.fromarray(colored_arr), binary_label))
                
            # Image.fromarray(colored_arr).save('./data/sample/{}.png'.format(idx))

        if not os.path.exists(colored_mnist_dir):
            os.makedirs(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
        torch.save(train3_set, os.path.join(colored_mnist_dir, 'train3.pt'))
        torch.save(test1_set, os.path.join(colored_mnist_dir, 'test1.pt'))
        torch.save(test2_set, os.path.join(colored_mnist_dir, 'test2.pt'))

class RandomColoring(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _generate_random_color(self):
        #生成一个随机的RGB颜色元组，值在[0, 1]之间
        return (random.random(), random.random(), random.random())

    def forward(self, tensor_image: torch.Tensor) -> torch.Tensor:
        #这个变换要在ToTensor之后做，注意：ToTensor会把颜色的三通道放在第一个维度！！
       
        #识别前景和背景，黑色背景中有值的就是数字
        #从通道检查，如果该位置任意个通道有值，那它就是数字所在的位置，这将返回一个28*28的掩码数组，表示对应位置有没有数字
        foreground_mask = ~torch.all(tensor_image > 0.9, dim=0)
        background_mask=~foreground_mask
        #print(foreground_mask)

        #生成随机的前景色和背景色
        foreground_color=self._generate_random_color()
        background_color=self._generate_random_color()
        #如果颜色太近重新生成
        while torch.linalg.norm(torch.tensor(foreground_color)-torch.tensor(background_color))< 0.5:
            background_color = self._generate_random_color()

        #先创建一个用背景色填充的图像
        new_image = torch.zeros_like(tensor_image)
        new_image[0,:, :] = background_color[0]
        new_image[1,:, :] = background_color[1]
        new_image[2,:, :] = background_color[2]
        
        #在新图像上使用掩码将有数字的位置设置为前景色
        new_image[0,foreground_mask] = foreground_color[0]
        new_image[1,foreground_mask] = foreground_color[1]
        new_image[2,foreground_mask] = foreground_color[2]
        
        return new_image





#图像预处理流程
#转换为 Tensor，以及归一化
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_transform=transforms.Compose([
    #transforms.Grayscale(num_output_channels=1), 
    #transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # 随机裁剪并缩放  
    #transforms.RandomGrayscale(p=0.1),       # 随机灰度化
    #degrees旋转角度（最大20度），translate平移的最大距离（宽度的0.1倍），shear剪切变换的角度范围
    #transforms.RandomAffine(degrees=20, translate=(0.1, 0.1),shear=10, scale=(0.9, 1.1)),  
    transforms.ToTensor(),
    RandomColoring(),
    #transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # 随机擦除，scale擦除区域的面积范围，ratio擦除区域的宽高比的范围
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

print("Loading")
train_dataset1=ColoredMNIST(root='./data', env='train1', transform=test_transform)
train_dataset2= ColoredMNIST(root='./data', env='train2', transform=test_transform)
train_dataset3= ColoredMNIST(root='./data', env='train3', transform=test_transform)
test_dataset1= ColoredMNIST(root='./data', env='test1', transform=train_transform)
test_dataset2= ColoredMNIST(root='./data', env='test2', transform=train_transform)

# 打印数据集大小，确认加载成功
print(f"Size of train_dataset_1: {len(train_dataset1)}")
print(f"Size of test_dataset_2: {len(test_dataset2)}")

print(train_dataset1)
see1=next(iter(train_dataset1))
see2=next(iter(test_dataset1))
print(see1[0].shape,see2[0].shape)


#把所有训练集搞在一起
all_train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])

BATCH_SIZE = 128
#dataloader
my_train_loader1 = DataLoader(dataset=train_dataset1, batch_size=BATCH_SIZE, shuffle=True)
my_train_loader2 = DataLoader(dataset=train_dataset2, batch_size=BATCH_SIZE, shuffle=True)
my_train_loader3 = DataLoader(dataset=train_dataset3, batch_size=BATCH_SIZE, shuffle=True)
my_test_loader1 = DataLoader(dataset=test_dataset1, batch_size=BATCH_SIZE, shuffle=False)
my_test_loader2 = DataLoader(dataset=test_dataset2, batch_size=BATCH_SIZE, shuffle=False)
all_dataloader=DataLoader(dataset=all_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Total training{len(all_train_dataset)}")
print(f"Test1 {len(test_dataset1)}")
print(f"Test2 {len(test_dataset2)}")

subset_indices = list(range(int(len(all_train_dataset) * 0.2))) 
# 创建数据子集
debug_train_dataset = Subset(all_train_dataset, subset_indices)
debug_dataloader = DataLoader(dataset=debug_train_dataset, batch_size=BATCH_SIZE, shuffle=True)



def denormalize(tensor):
    # 1. 从输入的tensor获取它所在的设备，这是最关键的一步
    device = tensor.device
    
    # 2. 在创建mean和std张量时，明确地将它们也放在同一个设备上
    mean = torch.tensor([0.5, 0.5, 0.5], device=device)
    std = torch.tensor([0.5, 0.5, 0.5], device=device)

    # 3. 使用广播机制进行计算，(C, H, W) -> (C, 1, 1)
    #    这样 (C,H,W) 的张量可以和 (C,1,1) 的张量安全运算
    denormalized_tensor = tensor * std[:, None, None] + mean[:, None, None]
    
    # 4. 将数值限制在[0, 1]范围
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
    
    return denormalized_tensor
import matplotlib.gridspec as gridspec
def show_images_3_then_2_centered(data_loaders, titles=None):
    """
    从5个数据加载器中各取一张图片，以"第一行3张，第二行2张且居中"的布局显示。
    """
    if len(data_loaders) != 5:
        print("错误：需要正好5个数据加载器。")
        return

    # --- 数据准备部分 (和之前一样) ---
    images_to_show = []
    labels_to_show = []
    for loader in data_loaders:
        try:
            images_batch, labels_batch = next(iter(loader))
            images_to_show.append(images_batch[0])
            labels_to_show.append(labels_batch[0])
        except StopIteration:
            print("一个数据加载器为空，无法获取图片。")
            return
            
    # --- 开始绘图 ---
    
    # 1. 创建一个画布
    fig = plt.figure(figsize=(10, 5)) # 调整画布大小以适应布局

    # 2. 定义一个 2x6 的高分辨率 GridSpec 网格
    gs = gridspec.GridSpec(2, 6, figure=fig)

    # 3. 创建子图并指定它们在网格中的位置 (使用切片)
    #    第一行 (3张图，铺满)
    ax1 = fig.add_subplot(gs[0, 0:2]) # 第0行，跨越第0和1列
    ax2 = fig.add_subplot(gs[0, 2:4]) # 第0行，跨越第2和3列
    ax3 = fig.add_subplot(gs[0, 4:6]) # 第0行，跨越第4和5列
    
    #    第二行 (2张图，居中)
    ax4 = fig.add_subplot(gs[1, 1:3]) # 第1行，跨越第1和2列 (左边留空)
    ax5 = fig.add_subplot(gs[1, 3:5]) # 第1行，跨越第3和4列 (右边留空)

    # 将所有子图对象和数据放入列表，方便循环处理
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    # 遍历每个子图和对应的数据
    for i in range(5):
        ax = axes[i]
        image_tensor = images_to_show[i]
        label = labels_to_show[i]

        # 逆标准化和格式转换
        image_tensor_denorm = denormalize(image_tensor.unsqueeze(0)).squeeze(0)
        np_image = image_tensor_denorm.permute(1, 2, 0).numpy()

        # 在子图上显示图片
        ax.imshow(np_image)
        
        # 设置标题
        title = f"DataLoader {i+1}\nLabel: {label.item()}"
        if titles and len(titles) == 5:
            title = f"{titles[i]}\nLabel: {label.item()}"
        ax.set_title(title, fontsize=12)
        
        # 关闭坐标轴
        ax.axis('off')
    
    # 调整子图间距，防止重叠
    fig.tight_layout()
    plt.show()

custom_titles = ["train1", "train2", "train3", "test1", "test2"]
show_images_3_then_2_centered([my_train_loader1,my_train_loader2,my_train_loader3,my_test_loader1,my_test_loader2],custom_titles)

print("Visualizing a batch from DataLoader:")
# show_batch(my_train_loader1)
# show_batch(my_test_loader1)
# show_batch(my_test_loader2)

#----------------------------------------------数据处理部分，复制即可-------------------------------

class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        #输入是 3x28x28，输出是类别数
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            #nn.InstanceNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            #nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    def forward(self, x):
        features = self.features(x)
        x = features.view(-1, 16 * 5 * 5) #展平操作
        logits = self.classifier(x)
        return logits,features
    
def test_model(model,test_loader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = model(images)
            _, predicted = torch.max(outputs.data, 1)#取最大概率作为结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy{accuracy:.2f} %')
    return accuracy


def irm_train(model, train_loaders,test_loader1,test_loader2, loss_fn, optimizer,scheduler, device, num_epochs=15, irm_lambda=0.3):
    
    print("\nStarting IRM training")
    print(f"Dataloader length:{len(train_loaders)}")
    history = {
        'train_loss': [],
        'train_acc':[],
        'rsc_loss':[],
        'irm_penalty': [],
        'test_acc1':[],
        'test_acc2':[],
        'test_avg':[]
        }
    train_loader_iters = [iter(cycle(loader)) for loader in train_loaders]
    num_batches_per_epoch = max([len(loader) for loader in train_loaders])
    for epoch in range(num_epochs):
        model.train()
        #超参数的动态优化部分
        #源代码的逻辑是每10个epoch更新一次pecent，pecent取消抑制的数量，高pecent只有置信度下降最剧烈的会被真正抑制
        interval = 6
        if epoch > 0 and epoch % interval == 0:
            model.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10
            print(f"Epoch {epoch+1}, self.pecent updated to: {model.pecent:.2f}")
        current_irm_lambda = irm_lambda
        if epoch < 10: # 预热期，IRM惩罚较小
            current_irm_lambda = irm_lambda * 0.01 
        elif epoch < 20: # 中期，IRM惩罚增加
            current_irm_lambda = irm_lambda * 0.1
        
        #一个epoch的参数
        running_loss = 0.0
        #n_batches = len(train_loaders)
        n_batches = max(len(loader) for loader in train_loaders)
        running_erm_loss = 0.0
        running_irm_penalty = 0.0
        running_acc=0.0

        for batch_idx in range(num_batches_per_epoch):
            optimizer.zero_grad()
            
            env_features_for_irm = []
            env_labels_for_irm = []
            batch_erm_loss_sum = 0.0
            total_train_batch = 0
            correct_train_batch = 0
            
            for loader_iter in train_loader_iters:
                #print(images.shape)
                images, labels = next(loader_iter)
                images = images.to(device)
                labels = labels.to(device)
                # 前向传播
                outputs,features = model(images)
                env_features_for_irm.append(features)
                env_labels_for_irm.append(labels)
                
                # 计算当前环境的ERM损失
                erm_loss_e = loss_fn(outputs, labels)
                batch_erm_loss_sum += erm_loss_e

                # 统计训练准确率
                _, predicted = torch.max(outputs.data, 1)
                total_train_batch += labels.size(0)
                correct_train_batch += (predicted == labels).sum().item()
            #------------------每个环境做完，下面进入batch层面----------------------------
            erm_loss = batch_erm_loss_sum / len(train_loaders)   
            # 计算 IRM 惩罚项
            irm_penalty = compute_irm_penalty(env_features_for_irm, env_labels_for_irm, model.classifier, device)
            # 总损失 = 经验风险 + IRM惩罚
            total_loss = erm_loss + current_irm_lambda * irm_penalty 
            total_loss.backward()
            optimizer.step()
            # 统计信息
            running_erm_loss += erm_loss.item()
            running_irm_penalty += irm_penalty.item() 
            running_loss+=total_loss/len(train_loaders)
            batch_acc=correct_train_batch/total_train_batch 
            running_acc+=batch_acc
            #--------------------每个batch做完-----------------------------
        # 计算epoch统计量
        avg_train_loss = running_loss / n_batches
        avg_train_acc = running_acc / n_batches
        avg_erm_loss=running_erm_loss/n_batches
        avg_irm_pen=running_irm_penalty/n_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss:{avg_train_loss:.4f}, "
              f"Train Acc:{100*avg_train_acc:.2f}%, "
              f"IRM_lambda:{current_irm_lambda}"
             )  
        test_acc1=test_model(model,test_loader1, device)
        test_acc2=test_model(model,test_loader2, device)
        #记录
        history['train_loss'].append(avg_train_loss.item())
        history['train_acc'].append(avg_train_acc)
        history['rsc_loss'].append(avg_erm_loss)
        history['irm_penalty'].append(avg_irm_pen)
        history['test_acc1'].append(test_acc1)
        history['test_acc2'].append(test_acc2)
        history['test_avg'].append((test_acc1 + test_acc2) / 2)
        if epoch%2==0:
            #visualize_attention_map(model, test_loader, my_device, image_index=10)
            pass
        
        #scheduler.step()
    print("Finished training")
    return history

def compute_irm_penalty(processed_features_list, labels_list, classifier_head, device):
    """
    计算基于 IRMv1 的梯度不变性惩罚
    processed_features_list: 每个环境经过 RSC 处理后的特征列表
    labels_list: 每个环境的标签列表。
    classifier_head: 模型的分类器部分(nn.Sequential)
    """
    if len(processed_features_list) < 2:
        return torch.tensor(0.0, device=device)

    #LeNet5的分类器输入是16*5*5
    feature_dim = processed_features_list[0].shape[1] * processed_features_list[0].shape[2] * processed_features_list[0].shape[3]
    num_classes = classifier_head[-1].out_features
    
    #虚拟权重，每次调用都重新创建
    dummy_w = torch.randn(num_classes, feature_dim, requires_grad=True, device=device)
    dummy_b = torch.randn(num_classes, requires_grad=True, device=device)

    penalty = 0.0
    for features_e, labels_e in zip(processed_features_list, labels_list):
        # 展平特征
        features_flat_e = torch.flatten(features_e, 1)
        #通过虚拟权重计算logits
        logits_e = F.linear(features_flat_e, dummy_w, dummy_b)
        # 计算该环境的损失
        loss_e = F.cross_entropy(logits_e, labels_e)
        # 计算损失相对于虚拟权重 dummy_w 的梯度
        # create_graph=True允许对梯度本身进行二次求导（即惩罚梯度）
        grad_w_e = grad(loss_e, dummy_w, create_graph=True)[0]
        # 计算梯度范数的平方
        penalty += grad_w_e.pow(2).sum()
        
    return penalty / len(processed_features_list) # 平均惩罚项

def get_optim_and_scheduler(network, epochs, lr=0.01, train_all=True, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=0.9, nesterov=nesterov, lr=lr)
    step_size = 5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,gamma=0.1)
    print("Step size: %d" % step_size)
    return optimizer, scheduler

NUM_EPOCHS=15
def train(model,train_loader,train_test1,train_test2,loss,optimizer,scheduler,device):
    print("\nStarting Normal training")
    print(f"Dataloader length: 1")
    history = {
    'train_loss': [],
    'train_acc': [],
    'test_acc1':[],
    'test_acc2':[],
    'test_avg':[]
    }
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss=0.0
        total_train=correct_train=0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            #outputs = model(images)
            outputs,features = model(images)
            loss1 = loss(outputs, labels)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            running_loss += loss1.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        #scheduler.step()
        print("comes test1")    
        test_acc1=test_model(model,train_test1, device)
        print("comes test2")    
        test_acc2=test_model(model,train_test2, device)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc1'].append(test_acc1)
        history['test_acc2'].append(test_acc2)
        history['test_avg'].append((test_acc1 + test_acc2) / 2)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}],Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print("Finished")
    return history

import math
import torch.nn.init as init
def initialize_lenet_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight) 
        if m.bias is not None:
            init.constant_(m.bias, 0)

class LeNet5RSC(nn.Module):

    def __init__(self, num_classes=2, pecent=0.1,drop_rate=0.05):
        super(LeNet5RSC, self).__init__()
        
        # LeNet5 的特征提取部分
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # LeNet5 的分类器部分
        self.class_classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
        self.num_classes=num_classes
        self.pecent=pecent 
        self.drop_rate=drop_rate
        # 应用初始化
        self.apply(initialize_lenet_weights)

    def forward(self, x, gt=None, flag=None, epoch=None):
        #x:128,3,28,28; gt:128
        raw_features = self.features_extractor(x) # 原始特征 (B, C, H, W)
        #raw_features:128,16,5,5
        if flag:
            #梯度和重要性计算
            #复制一份模型
            grad_model = self.__class__(num_classes=self.num_classes, pecent=self.pecent).to(x.device)
            grad_model.load_state_dict(self.state_dict())
            grad_model.train()
            x_new=grad_model.features_extractor(x).requires_grad_(True)#不要直接clone而是从复制的模型构建
            #x_new:128,16,5,5
            #x_new = raw_features.clone().detach().requires_grad_(True)
            x_new.retain_grad()
            x_new_view = x_new.view(x_new.size(0), -1)#LeNet5的分类器直接接收展平特征
            output = grad_model.class_classifier(x_new_view)
            #output:128,2
            class_num = output.shape[1]
            one_hot_sparse = torch.zeros_like(output).scatter_(1, gt.unsqueeze(1), 1)#遍历个行，列的选取按照gt，填充为1，这里构建128*2独热编码
            one_hot = torch.sum(output * one_hot_sparse)

            # 反向传播获取梯度
            grad_model.zero_grad()
            one_hot.backward(retain_graph=True) # retain_graph=True是必需的，因为后续还有计算
            grads_val = x_new.grad.clone().detach()
            #grads_val:128,16,5,5
            
            # 计算通道平均梯度和空间重要性
            num_rois, num_channel, H, W = x_new.shape
            HW = H * W #5*5=25
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            #grad_channel_mean:128,16,1
            #修改处，空间重要性计算，使用通道平均梯度乘以原始特征，修改后注释这两行
            channel_weights=torch.mean(grads_val, dim=[2, 3], keepdim=True)
            #channel_weights:128,16,1,1,它强调哪些通道是重要的
            spatial_mean_grad_cam = torch.sum(x_new.detach() * channel_weights, dim=1)
            #spatial_mean_grad_cam = F.relu(spatial_mean_grad_cam)
            spatial_mean = spatial_mean_grad_cam.view(num_rois, HW)
            #spatial_mean:128,25
            #修改处
            #spatial_mean = torch.mean(torch.abs(grads_val),dim=1).view(num_rois,HW)
            
            grad_model.zero_grad() #再次清零梯度，因为已经提取了需要的值
            
            #创建掩码矩阵
            #修改处：两者都要
            #choose_one = random.randint(0, 9)
            #if choose_one <= 4: # 50% 概率空间抑制
            spatial_drop_num = math.ceil(HW * self.drop_rate)
            th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
            th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW)
            
            mask_cuda = torch.where(spatial_mean > th18_mask_value, 
                                    torch.zeros_like(spatial_mean),
                                    torch.ones_like(spatial_mean))
            mask_space = mask_cuda.reshape(num_rois, H, W).view(num_rois, 1, H, W)
            #else: # 50% 概率通道抑制
            vector_thresh_percent = math.ceil(num_channel * self.drop_rate)
            vector_thresh_value = torch.sort(grad_channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
            vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
            
            vector = torch.where(grad_channel_mean > vector_thresh_value,
                                    torch.zeros_like(grad_channel_mean),
                                    torch.ones_like(grad_channel_mean))
            mask_chan = vector.view(num_rois, num_channel, 1, 1)
            mask_all=mask_chan*0.7+mask_space*0.3
            #mask_all=mask_space
            # 基于置信度变化的掩码修正
            # 1. 计算抑制前后的分类置信度
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = raw_features * mask_all 
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.class_classifier(x_new_view_after) # 使用主模型的分类器
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            # 2. 计算置信度下降程度
            one_hot_sparse_for_prob = torch.zeros_like(cls_prob_before).scatter_(1, gt.unsqueeze(1), 1)
            before_vector = torch.sum(one_hot_sparse_for_prob * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse_for_prob * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros_like(change_vector))
            
            # 3. 找到置信度下降不显著的样本
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent))]
            drop_index_fg = change_vector.gt(th_fg_value)
            ignore_index_fg = ~drop_index_fg
            #修改处
            # if ignore_index_fg.any():#至少有一个要修正
            #     #获取低影响组样本的置信度下降值
            #     change_vector_ignored = change_vector[ignore_index_fg]
            #     #下降值归一化
            #     normalized_change_ignored = change_vector_ignored / (th_fg_value + 1e-8)
            #     #当normalized_change_ignored接近0时，correction_factor接近1(完全修正)
            #     correction_factor=1.0 - normalized_change_ignored
            #     # d. 扩展修正因子以匹配 mask_all 的形状
            #     #    mask_all 的形状可能是 (B, 1, H, W) 或 (B, C, 1, 1)
            #     #    我们只需要对 ignore_index_fg 对应的样本进行操作
            #     correction_factor_expanded = correction_factor.view(-1, 1, 1, 1) # 形状 (N, 1, 1, 1), N是低影响样本数
            #     # e. 获取低影响组的原始掩码
            #     original_mask_ignored = mask_all[ignore_index_fg]
            #     # f. 计算修正后的新掩码 (线性插值)
            #     #    (1 - factor) * old_mask + factor * 1.0
            #     corrected_mask = (1 - correction_factor_expanded) * original_mask_ignored + correction_factor_expanded * 1.0
            #     # g. 将修正后的掩码放回 mask_all 中
            #     mask_all[ignore_index_fg] = corrected_mask
            # 4. 对这些样本取消抑制 (将它们的掩码设为 1)
            mask_all[ignore_index_fg, :] = 1

            #保持 mask_all 为不可求导，作为一种正则化
            del grad_model 
            processed_features = raw_features * mask_all.detach()
        else:
            processed_features = raw_features

        # 最终的前向传播
        x = processed_features.view(processed_features.size(0), -1)
        logits = self.class_classifier(x)
        
        return logits


def test_model_rsc(model,test_loader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)#取最大概率作为结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy{accuracy:.2f} %')
    return accuracy

def rsc_train(model, train_loaders,test_loader1,test_loader2,loss_fn, optimizer, scheduler,device, num_epochs=NUM_EPOCHS,use_rsc=True):
    current_step=0
    print("\nStarting RSC training")
    history = { 
    'train_loss': [],
    'train_acc': [],
    'test_acc1':[],
    'test_acc2':[],
    'test_avg':[]
    }
    for epoch in range(num_epochs):
        model.train()
        
        #源代码的逻辑是每10个epoch更新一次pecent，pecent取消抑制的数量，高pecent只有置信度下降最剧烈的会被真正抑制
        interval = 6
        if epoch > 0 and epoch % interval == 0:
            model.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10
            print(f"Epoch {epoch+1}, self.pecent updated to: {model.pecent:.2f}")
        
        running_loss = 0.0
        total_train = correct_train = 0
        n_batches = len(train_loaders)
        
        #print(n_batches)
        for images,labels in train_loaders:
            #print(images.shape)
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(images, gt=labels, flag=use_rsc, epoch=epoch)
            total_loss = loss_fn(outputs, labels) 
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss
            # 计算准确率(使用最后一个环境的输出)
            _,predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        print("comes to test")    
        test_acc1=test_model_rsc(model,test_loader1, device)
        test_acc2=test_model_rsc(model,test_loader2, device)
        history['test_acc1'].append(test_acc1)
        history['test_acc2'].append(test_acc2)
        history['test_avg'].append((test_acc1+test_acc2)/2.0)
        # 计算epoch统计量
        avg_train_loss = running_loss / n_batches
        train_acc = 100 * correct_train / total_train
        
        # 记录历史
        history['train_loss'].append(avg_train_loss.item())
        history['train_acc'].append(train_acc)
        if epoch<0:
        #visualize_attention_map(model, my_test_loader1, my_device, image_index=0)
            #visualize_attention_map(model, test_loader, my_device, image_index=10)
        #visualize_attention_map(model, my_test_loader1, my_device, image_index=100)
            pass
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Pecent{model.pecent}"
             )
        scheduler.step()
    print("Finished RSC training")
    return history

def compare_histories(histories, labels=None, metric='test_avg', colors=None):
    """
    histories: list of dict
    labels: list of str
    metric: str
    colors: list of str (比如 ['blue', 'red', 'green'])
    """
    plt.figure(figsize=(10,5))
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(histories))]
    if colors is None:
        # 使用一个默认颜色序列
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
    markers = ['o', 's', '^', 'D', 'X', '*']  # 提供不同 marker
    
    for i, (hist, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(hist[metric]) + 1)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(epochs, hist[metric], color=color, marker=marker, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Validating Accuracy(%)')
    #plt.title('Validating_Average_Accuracy(%)')
    plt.legend()
    plt.grid(True)
    plt.show()




my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {my_device}")
my_loss=nn.CrossEntropyLoss()

my_model_rsc=LeNet5RSC(num_classes=2).to(my_device)
my_optimizer_rsc,my_scheduler_rsc=get_optim_and_scheduler(my_model_rsc,NUM_EPOCHS)
my_history_rsc=rsc_train(my_model_rsc,all_dataloader,my_test_loader1,my_test_loader2,my_loss,my_optimizer_rsc,my_scheduler_rsc,my_device,use_rsc=True)
print(my_history_rsc)

my_model=LeNet5(num_classes=2).to(my_device)
my_optimizer,my_scheduler=get_optim_and_scheduler(my_model,NUM_EPOCHS)
my_history1=train(my_model,all_dataloader,my_test_loader1,my_test_loader2,my_loss,my_optimizer,my_scheduler,my_device)
print(my_history1)

my_model_irm=LeNet5(num_classes=2).to(my_device)
my_optimizer_irm,my_scheduler_irm=get_optim_and_scheduler(my_model_irm,NUM_EPOCHS)
my_history2=irm_train(my_model_irm,[my_train_loader1,my_train_loader2,my_train_loader3],my_test_loader1,my_test_loader2,my_loss,my_optimizer_irm,my_scheduler_irm,my_device)
print(my_history2)

compare_histories(
    [my_history1, my_history_rsc, my_history2],
    labels=['Baseline', 'RSC', 'IRM']
)