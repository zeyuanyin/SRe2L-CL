import argparse
import copy
import gc
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from tiny_imagenet import TinyImageNet
from torchvision.datasets import ImageFolder
from utils import ParamDiffAug, TensorDataset, evaluate

NUM_CLASS = 200


def create_model(model_name, path=None):
    # print("Creating resnet model for tiny-imagenet")
    model = torchvision.models.get_model(model_name, weights=None, num_classes=NUM_CLASS)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    if path is not None:
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "module." in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    model.cuda()
    return model


def main():
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--model", type=str, default="resnet18", help="model")
    parser.add_argument("--ipc", type=int, default=20, help="image(s) per class")
    parser.add_argument("--steps", type=int, default=5, help="5/10-step learning")
    parser.add_argument("--num_eval", type=int, default=1, help="evaluation number")
    parser.add_argument("--epoch_eval_train", type=int, default=100, help="epochs to train a model")
    parser.add_argument("--lr_net", type=float, default=0.01, help="learning rate for updating network parameters")
    parser.add_argument("--batch_train", type=int, default=256, help="batch size for training networks")
    parser.add_argument("--data_dir", type=str, default="./tiny_imagenet", help="Tiny-imagenet path")
    parser.add_argument("--train_dir", default=None, type=str, help="synthetic data directory")
    parser.add_argument("--teacher_model", default="resnet18", type=str, help="teacher model name")
    parser.add_argument("--teacher_path", default=None, type=str, help="teacher model checkpoint path")
    parser.add_argument("-T", "--temperature", default=1.0, type=float, help="temperature for distillation loss")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dsa_param = ParamDiffAug()
    args.dsa = True  # augment images for all methods
    args.dsa_strategy = "color_crop_cutout_flip_scale_rotate"

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, please run with a GPU")
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    # load/download tiny-imagenet data
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    data_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = ImageFolder(args.train_dir, transform=data_transform)
    test_dataset = TinyImageNet(args.data_dir, split="val", download=True, transform=data_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    print("loading all data from local disk to memory...")
    """ all training data """
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(NUM_CLASS)]

    mp.set_sharing_strategy("file_system")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=16)
    for images, labels in tqdm.tqdm(train_loader):
        images_all.append(images)
        labels_all.append(labels)
    images_all = torch.cat(images_all, dim=0).cuda()
    labels_all = torch.cat(labels_all, dim=0).cuda()
    # labels_all = torch.tensor(labels_all, dtype=torch.long).cuda()
    # labels_all = labels_all.clone().detach().type(torch.long).cuda()

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    def get_images(c, n):  # get n images from class c
        # idx_shuffle = np.random.permutation(indices_class[c])[:n]
        idx_shuffle = indices_class[c][:n]  # not random, get the first n images
        return images_all[idx_shuffle]

    # load teacher model
    teacher_model = create_model(args.teacher_model, args.teacher_path)
    for p in teacher_model.parameters():
        p.requires_grad = False

    results = np.zeros((args.steps, args.num_eval))
    num_classes_step = NUM_CLASS // args.steps
    np.random.seed(0)
    class_order = np.random.permutation(NUM_CLASS).tolist()
    print("\n", "=" * 32)
    print("class_order: ", class_order)

    images_train_all = []
    labels_train_all = []
    for step in range(args.steps):
        classes_current = class_order[step * num_classes_step : (step + 1) * num_classes_step]
        images_train_all += [torch.cat([get_images(c, args.ipc) for c in classes_current], dim=0)]
        labels_train_all += [
            torch.tensor([c for c in classes_current for i in range(args.ipc)], dtype=torch.long).cuda()
        ]

    for step in range(args.steps):
        print("\n", "-" * 32, f" step {step}")
        classes_seen = class_order[: (step + 1) * num_classes_step]
        print(f"classes_seen_num: {len(classes_seen)} out of {NUM_CLASS}, {len(classes_seen) / NUM_CLASS * 100:.2f}%")

        """ train data """
        images_train = torch.cat(images_train_all[: step + 1], dim=0).cuda()
        labels_train = torch.cat(labels_train_all[: step + 1], dim=0).cuda()

        """ test data """
        images_test = []
        labels_test = []
        for i in range(len(test_dataset)):
            lab = int(test_dataset[i][1])
            if lab in classes_seen:
                images_test.append(torch.unsqueeze(test_dataset[i][0], dim=0))
                labels_test.append(test_dataset[i][1])

        images_test = torch.cat(images_test, dim=0).cuda()
        labels_test = torch.tensor(labels_test, dtype=torch.long).cuda()
        dst_test_current = TensorDataset(images_test, labels_test)
        testloader = torch.utils.data.DataLoader(dst_test_current, batch_size=256, shuffle=False, num_workers=0)

        """ train model on the newest memory """
        accs = []
        for ep_eval in range(args.num_eval):
            net_eval = create_model(args.model)
            net_eval = net_eval.cuda()
            img_syn_eval = copy.deepcopy(images_train.detach())
            lab_syn_eval = copy.deepcopy(labels_train.detach())

            _, acc_train, acc_test = evaluate(
                ep_eval, net_eval, teacher_model, img_syn_eval, lab_syn_eval, testloader, args
            )
            del net_eval, img_syn_eval, lab_syn_eval
            gc.collect()  # to reduce memory cost
            accs.append(acc_test)
            results[step, ep_eval] = acc_test
        print("Evaluate %d x %s: mean = %.4f std = %.4f" % (args.num_eval, args.model, np.mean(accs), np.std(accs)))

    print("Done")
    print(results)

    results_str = ""
    for step in range(args.steps):
        results_str += "%.1f±%.1f  " % (np.mean(results[step]) * 100, np.std(results[step]) * 100)
    print(results_str, "\n\n")


if __name__ == "__main__":
    main()
