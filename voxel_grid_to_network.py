import pickle
import os
import numpy as np
from tqdm import trange

from run_nerf import config_parser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0'

class voxel_grid_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer_num, output_size):
        super(voxel_grid_MLP, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.middle = []
        self.middle = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for i in range(hidden_layer_num)])
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.input(x))
        for m_fc in self.middle:
            x = torch.relu(m_fc(x))
        x = self.output(x)
        return x

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels):
        self.point_clouds = torch.from_numpy(point_clouds).float().to(device)
        self.labels = torch.from_numpy(labels).long().to(device)

    def __getitem__(self, index):
        point_cloud = self.point_clouds[index]
        label = self.labels[index]
        return point_cloud, label

    def __len__(self):
        return len(self.labels)


def main():
    parser = config_parser()
    args = parser.parse_args()

    voxel_grid_objects = []
    with (open(os.path.join(args.datadir, "voxel_grid.pkl"), "rb")) as openfile:
        while True:
            try:
                voxel_grid_objects.append(pickle.load(openfile))
            except EOFError:
                break

    # Fetch voxel grid occupancy and world coordinates
    voxel_grid = voxel_grid_objects[0]['data']

    x_min, y_min, z_min = voxel_grid_objects[0]['min_point']
    x_max, y_max, z_max = voxel_grid_objects[0]['max_point']
    grid_size = voxel_grid_objects[0]['num_voxels']

    X, Y, Z = np.meshgrid(np.linspace(0, 1, grid_size),
                          np.linspace(0, 1, grid_size),
                          np.linspace(0, 1, grid_size))

    X = X * (x_max - x_min) + x_min
    Y = Y * (y_max - y_min) + y_min
    Z = Z * (z_max - z_min) + z_min
    pts = np.stack([X, Y, Z], axis=-1)

    # Define the point clouds and labels
    point_clouds = pts.reshape((-1, 3))
    labels = voxel_grid.reshape((-1)).astype(int)

    # Shuffle the data
    permutation = np.random.permutation(len(point_clouds))
    point_clouds = point_clouds[permutation]
    labels = labels[permutation]

    # Split the data into training and test sets
    train_size = int(0.8 * len(point_clouds))
    test_size = len(point_clouds) - train_size
    train_point_clouds, test_point_clouds = point_clouds[:train_size], point_clouds[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    # Create DataLoader objects for training and test sets
    bs = 160  # batch size
    train_dataset = PointCloudDataset(train_point_clouds, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataset = PointCloudDataset(test_point_clouds, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # Define hyperparameters
    input_size = 3  # Number of coordinates per point
    hidden_size = 64  # Number of neurons in the hidden layer
    hidden_layer_num = 2  # Number of hidden layers
    output_size = 2  # Number of labels
    lr = 0.001  # Learning rate
    num_epochs = 100  # Number of epochs
    i_accuracy = 10  # the frequency to evaluate the accuracy with the test dataset

    # Create MLP model
    model = voxel_grid_MLP(input_size, hidden_size, hidden_layer_num, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in trange(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

        # Evaluate the model
        if (epoch % i_accuracy == 0 and epoch > 0) or epoch == num_epochs - 1:
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy: %.2f%%' % (100 * correct / total))

    # Use the model to retrieve labels for specific point clouds
    test_cloud = torch.tensor([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]]).to(device)
    with torch.no_grad():
        outputs = model(test_cloud)
        _, predicted = torch.max(outputs.data, 1)
    print('Predicted labels:', predicted)

    # Save the model
    torch.save(model.state_dict(), args.datadir + '/voxel_grid_mlp.pth')


def test():
    input_size = 3  # Number of coordinates per point
    hidden_size = 64  # Number of neurons in the hidden layer
    hidden_layer_num = 2  # Number of hidden layers
    output_size = 2  # Number of labels
    net = voxel_grid_MLP(input_size, hidden_size, hidden_layer_num, output_size).to(device)
    net.load_state_dict(torch.load(os.path.join('data/Cuboid_cylinder_test/', 'voxel_grid_mlp.pth')))

    test_cloud = torch.tensor([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]).to(device)
    with torch.no_grad():
        outputs = net(test_cloud)
        _, pred = torch.max(outputs.data, 1)

    print("1")


if __name__ == '__main__':
    main()
    # test()