from fast import *
from model import *
from extra_utils import *
from engine import train_one_epoch, evaluate
import utils


if __name__ == '__main__':
    train_files_dir = 'WIDER_train/images'
    train_annotate_dir = 'WIDER_train_annotations'
    valid_files_dir = 'WIDER_val/images'
    valid_annotate_dir= 'WIDER_val_annotations'

    # use our dataset and defined transformations
    dataset = FaceImagesDataset(train_files_dir,train_annotate_dir, 480, 480, transforms= get_transform(train=True))
    dataset_test = FaceImagesDataset(valid_files_dir,valid_annotate_dir, 480, 480, transforms= get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    # indices_train = torch.randperm(len(dataset)).tolist()
    # indices_test = torch.randperm(len(dataset_test)).tolist()

    # train test split
    # tsize = int(len(dataset)*test_split)
    # dataset = torch.utils.data.Subset(dataset, indices_train)
    # dataset_test = torch.utils.data.Subset(dataset_test, indices_test)


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=10, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)



    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    num_classes = 2

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    num_epochs = 10

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

