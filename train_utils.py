import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import os
import time
import sys
import numpy as np
from equiv_tester import testForSymmetryRotationInvariance, testForAdditionInvariance
from equiv_net import EquivariantNet, load_checkpoint, save_checkpoint
from uuid import uuid4
from equiv_utils import flip
from equiv_aux_utils import random_4_rotation, random_8_flip_rotation, \
    flip_rotate_8, random_flip, flip_rotate_8_vectors_inv, random_8_flip_rotation_vec, \
    random_8_flip_rotation_square, flip_rotate_8_squares_inv, random_8_flip_rotation_line, \
    flip_rotate_8_lines_inv
from torch.utils.data import Dataset
from glob import glob


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, ims, labels, transform=None, double_aug=False, to_float=True):
        assert ims.shape[0] == labels.shape[0]
        self.ims = ims
        if ims.shape[-1] != 3:
            self.ims = ims.transpose(0, 2, 3, 1)
        if to_float:
            self.labels = labels.astype(np.float32)
        else:
            self.labels = labels.astype(np.int64)
        self.transform = transform
        self.double_aug = double_aug

    def __getitem__(self, index):
        x = self.ims[index]
        y = self.labels[index]
        if self.transform:
            if self.double_aug:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y

    def __len__(self):
        return self.labels.shape[0]

    def get_ones_proportion(self):
        return self.labels.mean()


class CustomDoubleTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, ims, labels, aug=[], normalize=True, vec_behaviour='vec'):
        assert ims.shape[0] == labels.shape[0]
        self.ims = ims.astype(np.float32)
        if ims.shape[-1] != 3 and ims.shape[-1] != 1:
            self.ims = ims.transpose(0, 2, 3, 1)
        self.labels = torch.tensor(labels.astype(np.float32))
        if normalize:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010))])
        else:
            self.transform = transforms.ToTensor()
        self.aug = aug
        if len(self.aug) > 0:
            self.double_transform = {'flip_rotate_vec': random_8_flip_rotation_vec,
                                     'flip_rotate_square': random_8_flip_rotation_square,
                                     'flip_rotate_line': random_8_flip_rotation_line}[aug[0] + '_' + vec_behaviour]
        else:
            self.double_transform = None

    def __getitem__(self, index):
        x = self.ims[index]
        y = self.labels[index]
        x = self.transform(x).float()

        if self.double_transform is not None:
            x, y = self.double_transform(x, y)
        return x, y

    def __len__(self):
        return self.labels.shape[0]


def get_data_transformer(dataAugmentation):
    name_to_transform = {'crop': transforms.RandomCrop(32, padding=4),
                         'flip': random_flip,
                         'rotate': random_4_rotation,
                         'flip_rotate': random_8_flip_rotation,
                         'flip_rotate_vec': random_8_flip_rotation_vec}
    torch_transforms = ['flip', 'rotate', 'flip_rotate', 'flip_rotate_vec']
    transform_list = []
    for aug in dataAugmentation:
        if aug not in torch_transforms:
            transform_list.append(name_to_transform[aug])
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    for aug in dataAugmentation:
        if aug in torch_transforms:
            transform_list.append(name_to_transform[aug])
    return transforms.Compose(transform_list)


def defineDataLoaders(dataset, trainBatchSize, testBatchSize, dataAugmentation):
    subnamed_datasets = ['Sun', 'Squares', 'Lines']
    loaders = {"Cifar-10": getCifar10DataLoaders,
               "Cifar-100": getCifar100DataLoaders,
               "Squares": getSquaresDataLoaders,
               'Lines': getLinesDataLoaders}
    for subname in subnamed_datasets:
        if subname in dataset:
            return loaders[subname](trainBatchSize, testBatchSize, dataAugmentation, '_'.join(dataset.split('_')[1:]))
    return loaders[dataset](trainBatchSize, testBatchSize, dataAugmentation)


def getSquaresDataLoaders(trainBatchSize, testBatchSize, dataAugmentation, subname):
    dataset_subtype, num_elems = subname.split('_')[:2]
    planes_dir = f'/content/drive/MyDrive/PytorchExperiments/squares_{dataset_subtype}'
    # planes_dir = f'/home/slavko/Documents/conferences/iclr2022/experiments/SQUARES/{dataset_subtype}'
    # transform_train = get_data_transformer(dataAugmentation)
    # transform_test = get_data_transformer([])

    test_ims_path = glob(planes_dir + '/test_ims_*.npy')[0]
    test_vecs_path = glob(planes_dir + '/test_vecs_*.npy')[0]

    train_ims = np.load(planes_dir + f'/train_ims_{num_elems}.npy')
    train_labels = np.load(planes_dir + f'/train_vecs_{num_elems}.npy')
    test_ims = np.load(test_ims_path)
    test_labels = np.load(test_vecs_path)

    # double_aug = ('vec' in dataAugmentation[0])

    trainset = CustomDoubleTensorDataset(train_ims, train_labels, aug=dataAugmentation, normalize=False,
                                         vec_behaviour='square')
    traintestset = CustomDoubleTensorDataset(train_ims, train_labels, aug=[], normalize=False, vec_behaviour='square')
    testset = CustomDoubleTensorDataset(test_ims, test_labels, aug=[], normalize=False, vec_behaviour='square')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize, shuffle=True)
    train_test_loader = torch.utils.data.DataLoader(traintestset, batch_size=trainBatchSize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=testBatchSize, shuffle=False)
    classes = ['not_plane', 'plane']
    shape = (1, 64, 64)
    return train_loader, train_test_loader, test_loader, classes, shape


def getLinesDataLoaders(trainBatchSize, testBatchSize, dataAugmentation, subname):
    dataset_subtype, num_elems = subname.split('_')[:2]
    planes_dir = f'/content/drive/MyDrive/PytorchExperiments/lines_{dataset_subtype}'
    # planes_dir = f'/home/slavko/Documents/conferences/iclr2022/experiments/LINES/{dataset_subtype}'
    # transform_train = get_data_transformer(dataAugmentation)
    # transform_test = get_data_transformer([])

    test_ims_path = glob(planes_dir + '/test_ims_*.npy')[0]
    test_vecs_path = glob(planes_dir + '/test_vecs_*.npy')[0]

    train_ims = np.load(planes_dir + f'/train_ims_{num_elems}.npy')
    train_labels = np.load(planes_dir + f'/train_vecs_{num_elems}.npy')
    test_ims = np.load(test_ims_path)
    test_labels = np.load(test_vecs_path)

    # double_aug = ('vec' in dataAugmentation[0])

    trainset = CustomDoubleTensorDataset(train_ims, train_labels, aug=dataAugmentation, normalize=False,
                                         vec_behaviour='line')
    traintestset = CustomDoubleTensorDataset(train_ims, train_labels, aug=[], normalize=False, vec_behaviour='line')
    testset = CustomDoubleTensorDataset(test_ims, test_labels, aug=[], normalize=False, vec_behaviour='line')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize, shuffle=True)
    train_test_loader = torch.utils.data.DataLoader(traintestset, batch_size=trainBatchSize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=testBatchSize, shuffle=False)
    classes = ['not_plane', 'plane']
    shape = (1, 64, 64)
    return train_loader, train_test_loader, test_loader, classes, shape



# import, load and normalize CIFAR
def getCifar10DataLoaders(trainBatchSize, testBatchSize, dataAugmentation):
    cuda_aval = torch.cuda.is_available()

    transform_train = get_data_transformer(dataAugmentation)
    transform_test = get_data_transformer([])
    # transform_train = None
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # if (dataAugmentation):
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    # else:
    #     transform_train = transform_test

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize,
                                              shuffle=True, num_workers=4, pin_memory=True)

    train_test_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform_test)
    train_test_loader = torch.utils.data.DataLoader(train_test_set, batch_size=trainBatchSize,
                                                    shuffle=False, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=testBatchSize,
                                             shuffle=False, num_workers=4, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    shape = (3, 32, 32)
    return trainloader, train_test_loader, testloader, classes, shape


def getCifar100DataLoaders(trainBatchSize, testBatchSize, dataAugmentation):
    cuda_aval = torch.cuda.is_available()

    transform_train = get_data_transformer(dataAugmentation)
    transform_test = get_data_transformer([])

    # transform_train = None
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    # ])
    #
    #
    # train_tr_list = []
    # for aug in dataAugmentation:
    #
    # if (dataAugmentation):
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    #     ])
    # else:
    #     transform_train = transform_test

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize,
                                              shuffle=True, num_workers=4, pin_memory=True)

    train_test_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                   download=True, transform=transform_test)
    train_test_loader = torch.utils.data.DataLoader(train_test_set, batch_size=trainBatchSize,
                                                    shuffle=False, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=testBatchSize,
                                             shuffle=False, num_workers=4, pin_memory=True)

    classes = tuple([str(i) for i in range(100)])
    shape = (3, 32, 32)
    return trainloader, train_test_loader, testloader, classes, shape


def testPerformance(net, dataLoader, dataName, average_flipped=False, average_flip_rotated=False, dataset=None):
    correct = 0
    total = 0
    loss = 0
    testCriterion = net.getLoss(reduction='sum')
    cuda_available = torch.cuda.is_available()

    with torch.no_grad():
        net.eval()
        for data in dataLoader:
            inputs, labels = data
            if cuda_available:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            labels_converted = net.convertLabels(labels)

            if average_flip_rotated:
                backward_rotation_function = {'Sun': flip_rotate_8_vectors_inv,
                                              'Squares': flip_rotate_8_squares_inv,
                                              'Lines': flip_rotate_8_lines_inv}[dataset.split('_')[0]]
                extended_inputs = flip_rotate_8(inputs)
                extended_outputs = [net(inp) for inp in extended_inputs]
                if net.tag is not None:
                    extended_outputs = backward_rotation_function(extended_outputs)
                outputs = sum(extended_outputs) / 8
            else:
                outputs = net(inputs)
                if average_flipped:
                    flipped_inputs = flip(inputs)
                    flipped_outputs = net(flipped_inputs)
                    outputs = 0.5 * (outputs + flipped_outputs)

            loss += testCriterion(outputs, labels_converted)
            total += labels.size(0)
            if net.tag is None:
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            else:
                outputs_norm = torch.sqrt((outputs * outputs).sum(axis=1, keepdims=True) + 1e-8)
                outputs_norm = outputs / outputs_norm
                labels_norm = torch.sqrt((labels * labels).sum(axis=1, keepdims=True) + 1e-8)
                labels_norm = labels / labels_norm
                dot_prod = (outputs_norm * labels_norm).sum(axis=1)
                dot_prod[dot_prod > 1] = 1.0
                dot_prod[dot_prod < -1] = -1.0

                angles = torch.acos(dot_prod)
                correct += angles.sum()
    loss = loss / total
    if net.tag is None:
        acc = 100 * correct / total
        print('Loss and accuracy on the %s set: %.5f, %.2f %%' % (dataName, loss, acc))
    else:
        acc = correct / total
        acc = acc / np.pi * 180
        print('Loss and avg angle on the %s set: %.5f, %.2f deg' % (dataName, loss, acc))
    return loss, acc


def adjust_learning_rate(optimizer, new_learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_learning_rate


def trainNetwork(net, learning_rate, lr_decay, numEpochs, ALPHA_CHANGING, MIN_ALPHA, MAX_ALPHA,
                 trainloader, train_test_loader, testloader, save_path=None, save_freq=None,
                 test_flipped=False, test_8=False, get_all_losses=False, dataset=None):
    # Train on training set
    cuda_avail = torch.cuda.is_available()

    weight = None
    if 'Reface' in dataset:
        proportion = trainloader.bin_proportion
        weight = torch.tensor([1 / (1 - proportion), 1 / proportion]).float()
        if cuda_avail:
            weight = weight.cuda()

    criterion = net.getLoss(reduction='mean', weight=weight)

    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 5e-4)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)

    if cuda_avail:
        net.cuda()

    net.printNet()
    print("Number of epochs: ", numEpochs)

    flag = 0
    alpha = 10 ** (MIN_ALPHA)

    all_results = []

    trainTimeStart = time.time()
    for epoch in range(numEpochs):  # loop over the dataset multiple times
        timeStart = time.time()
        # scheduler.step()
        net.train()
        running_loss = 0.0
        numBatches = 0
        if (ALPHA_CHANGING):
            alpha = 10 ** ((epoch / numEpochs) * (MAX_ALPHA - MIN_ALPHA) + MIN_ALPHA)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.float()

            if cuda_avail:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

            # labels_converted = net.convertLabels(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if (ALPHA_CHANGING):
                outputs = net(inputs, alpha)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, labels)

            '''if (flag == 0):
                save('s.dot', loss.grad_fn)
                flag = 1'''

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            numBatches += 1
        timeElapsed = time.time() - timeStart

        learning_rate *= lr_decay
        adjust_learning_rate(optimizer, learning_rate)

        print('[%d] loss: %.3f LR: %.5f Epoch time: %.2f s, Remaining time: %.2f s alpha: %.2f' %
              (epoch + 1, running_loss / numBatches, learning_rate, timeElapsed, (numEpochs - epoch - 1) * timeElapsed,
               alpha))

        if ((epoch + 1) % save_freq == 0):
            if get_all_losses:
                curr_result = []
            train_loss, train_acc = testPerformance(net, train_test_loader, "train")
            test_loss, test_acc = testPerformance(net, testloader, "test")
            if get_all_losses:
                curr_result.append(((train_loss.item(), train_acc.item()), (test_loss.item(), test_acc.item())))
            if test_flipped:
                print("Flip averaging: ")
                testPerformance(net, train_test_loader, "train", average_flipped=True, average_flip_rotated=False,
                                dataset=dataset)
                testPerformance(net, testloader, "test", average_flipped=True, average_flip_rotated=False,
                                dataset=dataset)
            if test_8:
                print("8 averaging: ")
                train_loss, train_acc = testPerformance(net, train_test_loader, "train", average_flipped=False,
                                                        average_flip_rotated=True, dataset=dataset)
                test_loss, test_acc = testPerformance(net, testloader, "test", average_flipped=False,
                                                      average_flip_rotated=True, dataset=dataset)
                if get_all_losses:
                    curr_result.append(((train_loss.item(), train_acc.item()), (test_loss.item(), test_acc.item())))
            if get_all_losses:
                all_results.append(curr_result)
        if save_path is not None and save_freq is not None:
            if (epoch + 1) % save_freq == 0:
                save_checkpoint(net, save_path[:-4] + f'_{epoch}.pth')
                print("checkpoint saved at ", save_path)

    trainDuration = time.time() - trainTimeStart
    print('Finished Training')
    train_loss, train_acc = testPerformance(net, train_test_loader, "train")
    test_loss, test_acc = testPerformance(net, testloader, "test")
    if get_all_losses:
        print("All losses: ")
        print(all_results)
    return train_loss, train_acc, test_loss, test_acc, trainDuration / numEpochs, all_results


def calculateDistributionOverClasses(net, testloader, classes):
    # Distribution over classes in test set
    if (len(classes) > 10):
        return
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    numClasses = len(classes)
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    cuda_available = torch.cuda.is_available()
    classDistr = torch.zeros(numClasses, numClasses)
    classTotal = torch.zeros(numClasses)
    with torch.no_grad():
        net.eval()
        for data in testloader:
            inputs, labels = data
            if cuda_available:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            # c = (predicted == labels).squeeze()
            for i in range(predicted.shape[0]):
                label = labels[i]
                pred = predicted[i]
                classDistr[label][pred] += 1
                classTotal[label] += 1
                # class_correct[label] += c[i].item()
                # class_total[label] += 1
    classDistr = 100 * classDistr / (classTotal.view(-1, 1))
    formatStr = "{:.2f}%\t" * numClasses

    print(("{}\t" * (1 + numClasses)).format(" ", *classes))
    for i, label in enumerate(classes):
        rowStr = label + ":\t" + formatStr.format(*classDistr[i].tolist())
        print(rowStr)


def trainNetworks(datasets, trainBatchSize, testBatchSize, dataAugmentations, netNames, learning_rate, lr_decay,
                  numEpochs, ALPHA_CHANGING, MIN_ALPHA, MAX_ALPHA, testRotation=True, testAddition=False,
                  save_folder=None, save_freq=None, test_flipped=False, test_8=[], get_all_losses=False):
    results = []
    param_num = []
    assert len(netNames) == len(datasets) == len(dataAugmentations) == len(test_8)
    for netName, dataset, dataAugmentation, test_8_instance in zip(netNames, datasets, dataAugmentations, test_8):
        trainloader, train_test_loader, testloader, classes, shape = defineDataLoaders(dataset, trainBatchSize,
                                                                                       testBatchSize, dataAugmentation)
        print("training ", netName)
        print("dataset: ", dataset)
        print("augmentation: ", dataAugmentation)
        if save_folder is not None:
            model_name = uuid4()
            model_path = os.path.join(save_folder, f'{model_name}.pth')
            print("Model name: ", model_name)
        netName = netName[:-1] + ", initDepth = " + str(shape[0]) + ", numClasses = " + str(len(classes)) + ")"
        command = "global net; net = " + netName
        exec(command)
        net.netName = netName
        param_num.append(net.numTrainableParams())
        res = trainNetwork(net, learning_rate, lr_decay, numEpochs, ALPHA_CHANGING, MIN_ALPHA, MAX_ALPHA, trainloader,
                           train_test_loader, testloader, save_path=model_path, save_freq=save_freq,
                           test_flipped=test_flipped, test_8=test_8_instance, get_all_losses=get_all_losses,
                           dataset=dataset)
        if net.tag is None:
            calculateDistributionOverClasses(net, testloader, classes)
        inv = ("", "")
        if testRotation:
            inv = testForSymmetryRotationInvariance(net, shape, len(classes))
        results.append(res + inv)
        if testAddition:
            testForAdditionInvariance(net)
        print(netName + ": training is finished!")

    print(" \n Overall results: (train_loss, train_acc, test_loss, test_acc)")
    print("Dataset: ", dataset)
    print("Num of epochs: ", numEpochs)
    print("Learning rate %.4f, l.r. decay: %.3f" % (learning_rate, lr_decay))
    print("Minibatch size: ", trainBatchSize)
    print("\n")
    for netName, result, params, dataAugmentation, dataset in zip(netNames, results, param_num, dataAugmentations, datasets):
        # print(netName + ": " + result)
        print("%s: %d params" % (netName, params))
        print("Augmentation: ", dataAugmentation)
        print("Dataset: ", dataset)
        print("%.5f, %.2f%%, %.5f, %.2f%%" % (result[0], result[1], result[2], result[3]))
        print(result[5])
        print("Average epoch duration: %.2f s" % result[4])
        if testRotation:
            print("Flip invariance: ", result[5])
            print("Rotation invariance: ", result[6])
        print("\n")
