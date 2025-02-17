# Split into training and testing sets
train_imgs = data[:60000]
train_labels = target[:60000]
    ax.imshow(train_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Label: {}".format(train_labels[i]))  # Use slicing train_labels[:10] to get first 10 labels
train_imgs = torch.from_numpy(train_imgs).float()
train_labels = torch.from_numpy(train_labels)
train_imgs = train_imgs / 255. * 2 - 1
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(f'Training with learning rate: {lr}')
    # Define loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim={hidden_dim}, num_layers=1')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 98%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim=100, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim=100, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim={hidden_dim}, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
train_imgs = train_imgs.view(-1, 28*28)
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f'Training with num_blocks={num_blocks}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
train_imgs = train_imgs.view(-1, 1, 28, 28)
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f'Training with num_blocks={num_blocks}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
train_imgs = train_imgs.view(-1, 1, 28, 28)
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f'Training with width multiplier={width_mult}, depth={depth}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
# Split into training and testing sets
train_imgs = data[:60000]
train_labels = target[:60000]
    ax.imshow(train_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Label: {}".format(train_labels[i]))  # Use slicing train_labels[:10] to get first 10 labels
train_imgs = torch.from_numpy(train_imgs).float()
train_labels = torch.from_numpy(train_labels)
train_imgs = train_imgs / 255. * 2 - 1
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(f'Training with learning rate: {lr}')
    # Define loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim={hidden_dim}, num_layers=1')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim=100, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim=100, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim={hidden_dim}, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim={hidden_dim}, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
train_imgs = train_imgs.view(-1, 28*28)
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f'Training with num_blocks={num_blocks}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
train_imgs = train_imgs.view(-1, 1, 28, 28)
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f'Training with num_blocks={num_blocks}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
train_imgs = train_imgs.view(-1, 1, 28, 28)
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f'Training with width multiplier={width_mult}, depth={depth}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
# Split into training and testing sets
train_imgs = data[:60000]
train_labels = target[:60000]
    ax.imshow(train_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Label: {}".format(train_labels[i]))  # Use slicing train_labels[:10] to get first 10 labels
train_imgs = torch.from_numpy(train_imgs).float()
train_labels = torch.from_numpy(train_labels)
train_imgs = train_imgs / 255. * 2 - 1
train_dataset = TensorDataset(train_imgs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(f'Training with learning rate: {lr}')
    # Define loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim={hidden_dim}, num_layers=1')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim=100, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')
    print(f'Training with hidden_dim={hidden_dim}, num_layers={num_layers}')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Training
        for images, labels in train_loader:
            optimizer.zero_grad()
            optimizer.step()
        # Stop training if accuracy reaches 95%
            print(f'Stopping training after reaching {accuracy}% accuracy.')