test_imgs = data[60000:]
test_labels = target[60000:]
test_imgs = torch.from_numpy(test_imgs).float()
test_labels = torch.from_numpy(test_labels)
test_imgs = test_imgs / 255. * 2 - 1
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Input to Hidden) vs Accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Hidden to Output) vs Accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_epochs_vs_accuracy.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 98:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed depth, vary width)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Input to Hidden) vs Accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Hidden to Output) vs Accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varywidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varywidthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Vary both width and depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthandwidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthandwidthaccuracy_over_epochs.png")
test_imgs = test_imgs.view(-1, 28*28)
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
test_imgs = test_imgs.view(-1, 1, 28, 28)
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/Resvarydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESvarydepthaccuracy_over_epochs.png")
test_imgs = test_imgs.view(-1, 1, 28, 28)
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Vary depth and width)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESvarydepthandwidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESvarydepthandwidthaccuracy_over_epochs.png")
test_imgs = data[60000:]
test_labels = target[60000:]
test_imgs = torch.from_numpy(test_imgs).float()
test_labels = torch.from_numpy(test_labels)
test_imgs = test_imgs / 255. * 2 - 1
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 98:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Input to Hidden) vs Accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Hidden to Output) vs Accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_epochs_vs_accuracy.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 98:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed depth, vary width)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Input to Hidden) vs Accuracy')
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption (Hidden to Output) vs Accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varywidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varywidthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
img_without_skip = mpimg.imread("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 99:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Vary both width and depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthandwidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthandwidthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Vary both width and depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/weightpruning_energy_vs_accuracy.png")
test_imgs = test_imgs.view(-1, 28*28)
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
test_imgs = test_imgs.view(-1, 1, 28, 28)
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/Resvarydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESvarydepthaccuracy_over_epochs.png")
test_imgs = test_imgs.view(-1, 1, 28, 28)
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Vary depth and width)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESvarydepthandwidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESvarydepthandwidthaccuracy_over_epochs.png")
test_imgs = data[60000:]
test_labels = target[60000:]
test_imgs = torch.from_numpy(test_imgs).float()
test_labels = torch.from_numpy(test_labels)
test_imgs = test_imgs / 255. * 2 - 1
test_dataset = TensorDataset(test_imgs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1)
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_epochs_vs_accuracy.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed depth, vary width)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varywidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varywidthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Fixed width, vary depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthaccuracy_over_epochs.png")
    best_accuracy = 0
        # Test
            for images, labels in test_loader:
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy}%')
        if accuracy >= 95:
    plt.xlabel('Accuracy')
    plt.title('Cumulative Energy Consumption vs Accuracy (Vary both width and depth)')
    plt.xlabel('Accuracy')
    plt.title('Epoch vs accuracy')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthandwidth_energy_vs_accuracy.png")
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/varydepthandwidthaccuracy_over_epochs.png")