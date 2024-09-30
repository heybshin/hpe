from detr.models.detr import MLP
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


if __name__ == '__main__':

    mlp = MLP(2, 368, 8, 3).cuda()
    print(mlp)
    epoch = 2000
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    for e in range(epoch + 1):
        data, label = torch.rand((100, 2), device='cuda'), \
            torch.zeros((100, 8), dtype=torch.float, device='cuda')

        data[:, 0] = data[:, 0] * 3
        data[:, 1] = data[:, 1] * 5

        indices = torch.floor(data).long()
        indices[:, 1] += 3
        for i in range(100):
            for j in range(2):
                label[i, indices[i, j]] = 1

        output = mlp(data)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            print(f'Epoch {e}/{epoch}, Loss: {loss.item()}')
            with torch.no_grad():
                data, label = torch.rand((100, 2), device='cuda'), \
                    torch.zeros((100, 8), dtype=torch.float, device='cuda')

                data[:, 0] = data[:, 0] * 3
                data[:, 1] = data[:, 1] * 5

                indices = torch.floor(data).long()
                indices[:, 1] += 3
                for i in range(100):
                    for j in range(2):
                        label[i, indices[i, j]] = 1

                output = mlp(data)
                pred = torch.sigmoid(output).cpu().numpy()
                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0
                print(f'F1 Score: {f1_score(label.cpu().numpy(), pred, average="macro")}')
                print(f'Accuracy: {accuracy_score(label.cpu().numpy(), pred)}')
                print(f'ROC AUC: {roc_auc_score(label.cpu().numpy(), pred)}')
    torch.save(mlp.state_dict(), '../outputs/mlp.pth')

