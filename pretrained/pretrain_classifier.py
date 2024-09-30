from utils.utils import fetch_epoch_data
import numpy as np
from braindecode.models import EEGNetv4
import torch
from torch.utils.data import DataLoader, TensorDataset
# cross validation
from sklearn.model_selection import KFold


def create_dataset(subject):

    if subject == 'lidan':
        c_path = 'data/eeg_data/lidan_referenced_dcrm_chanrm epochsC pruned with ICA.set'
        p_path = 'data/eeg_data/lidan_referenced_dcrm_chanrm epochsP pruned with ICA.set'
        # be_path = 'data/lidan_referenced_dcrm_chanrm epochsBE pruned with ICA.set'
        be_path = None
    elif subject == 'kds':
        c_path = 'data/eeg_data/kds_referenced_dcrm_chanrm epochsC pruned with ICA.set'
        p_path = 'data/eeg_data/kds_referenced_dcrm_chanrm epochsP pruned with ICA.set'
        be_path = None
    elif subject == 'p004':
        c_path = '../data/eeg_data/sub-P004jhjin_ses-S001_task-Bird_run-001_eeg pruned with ICA epochsC.set'
        p_path = '../data/eeg_data/sub-P004jhjin_ses-S001_task-Bird_run-001_eeg pruned with ICA epochsP.set'
        be_path = None
    elif subject == 'p005':
        c_path = '../data/eeg_data/sub-P005yjk_ses-S001_task-Bird_run-001_eeg pruned with ICA epochsC.set'
        p_path = '../data/eeg_data/sub-P005yjk_ses-S001_task-Bird_run-001_eeg pruned with ICA epochsP.set'
        be_path = None

    p_FCz_mean = fetch_epoch_data(p_path, event_id='P/P')
    c_FCz_mean = fetch_epoch_data(c_path)
    event_dict = {'Normal': p_FCz_mean, 'ErrP': c_FCz_mean}
    if be_path is not None:
        be_FCz_mean = fetch_epoch_data(be_path)
        event_dict['BE'] = be_FCz_mean

    data, labels = None, None
    for key, value in event_dict.items():

        if key == 'Normal':
            # choose a random part of the data
            # only for normal data, because there are too many normal data, to reduce the size of the dataset,
            # we randomly choose 300 samples, not that imbalanced
            idx = np.random.randint(0, value.shape[0], 300)
            value = value[idx]
            label = np.zeros((value.shape[0], 1))
        else:
            label = np.ones((value.shape[0], 1))

        if data is None:
            data = value
        else:
            data = np.concatenate((data, value), axis=0)

        if labels is None:
            labels = label
        else:
            labels = np.concatenate((labels, label), axis=0)

    return data, labels


if __name__ == '__main__':

    # subject = 'lidan'
    # subject = 'kds'
    # subject = 'p004'
    subject = 'p005'
    data, labels = create_dataset(subject=subject)

    # cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    avg_acc = 0
    for train_index, test_index in kf.split(data):

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # create the dataset
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        # create the dataloader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # create the model
        model = EEGNetv4(in_chans=1, n_classes=2, n_times=750, final_conv_length='auto')
        model = model.cuda()
        # create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        # create the loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # epoch
        for epoch in range(200):
            loss_all = 0
            for i, (X, y) in enumerate(dataloader):
                X, y = X.cuda(), y.cuda().squeeze()
                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                loss_all += loss.item()

                loss.backward()
                optimizer.step()
            # print the loss with 2 decimal places
            print(f'Epoch: {epoch}, Loss: {loss_all / len(dataloader):.4f}')

        # create the test dataset
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        # create the test dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        # test the model
        model.eval()
        correct = 0
        total = 0
        for i, (X, y) in enumerate(test_dataloader):
            X, y = X.cuda(), y.cuda().squeeze()
            y_pred = model(X)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f'Accuracy: {correct / total}')
        avg_acc += correct / total

    print(f'Average accuracy: {avg_acc / 5}')

    torch.save(model.state_dict(), 'outputs/pretrained_classifier/pretrained_classifier_{}.pth'.format(subject))
