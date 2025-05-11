from model import *
import os

if __name__ == "__main__":
    augment_everytime = True
    batch_size = 64
    training_parameters = [
        [1, 3, 64, 128],
        [2, 3, 64, 128]
    ]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    num_cores = os.cpu_count()
    if augment_everytime:
        train_dataset = PairedDSVariableEpoch('db/dataset_preprocessed/train/input', transform=transform)
        val_dataset = PairedDSVariableEpoch('db/dataset_preprocessed/val/input', transform=transform)
        test_dataset = PairedDSVariableEpoch('db/dataset_preprocessed/test/input', transform=transform)
    else:
        train_dataset = PariedImages('db/dataset_preprocessed/train', transform=transform)
        val_dataset = PariedImages('db/dataset_preprocessed/val', transform=transform)
        test_dataset = PariedImages('db/dataset_preprocessed/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cores - 2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cores - 2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cores - 2, pin_memory=True)

    inputs, outputs = next(iter(train_loader))
    print(inputs.shape, outputs.shape)
    print(inputs[1][1].max())
    modelpath =  os.path.join(os.getcwd(), 'models')
    historypath = os.path.join(os.getcwd(), 'history')


    check_and_make(modelpath)
    check_and_make(historypath)

    show_pair(inputs, outputs, 3)
    i = 0
    print('----TRAINING MODELS----')
    print('')
    for training_parameter in training_parameters:
        check_model_parameters(*training_parameter, 'unet')

    for loss, loss_name in zip([focal_frequency_loss, mse_loss], ['focal', 'mse']):
        for training_parameter in training_parameters:
            i += 1
            savename = 'model_{}_{}_loss_{}stacks_{}colors_{}Csize_{}Zsise'.format(i, loss_name, *training_parameter)
            print(' TRAINING NO {}'.format(i))
            print('Parameters:')
            print('Loss fcn: {}'.format(loss_name))
            print('Stacks: {}'.format(training_parameter[0]))
            print('C_size: {}'.format(training_parameter[2]))
            print('Z_size: {}'.format(training_parameter[3]))

            model = ChainedAutoencoder(*training_parameter, 'unet')
            model, history = train_model(model, loss, train_loader, val_loader, 45)

            model_savename = os.path.join(modelpath, '{}.pth'.format(savename))
            history_savename = os.path.join(historypath, '{}.csv'.format(savename))

            torch.save(model.state_dict(), model_savename)
            df = pd.DataFrame.from_dict(history)
            df.insert(0, 'epoch', df.index + 1)
            df.to_csv(history_savename, index = False)