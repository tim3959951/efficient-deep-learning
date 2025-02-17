final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumption_input_to_hidden = 0
    cumulative_energy_consumption_hidden_to_output = 0
    energy_consumptions_input_to_hidden = []
    energy_consumptions_hidden_to_output = []
        energy_consumption = 0
        energy_consumption_input_to_hidden = 0 
        energy_consumption_hidden_to_output = 0 
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
            energy_consumption_input_to_hidden += weight_updates[0].abs().sum().item() + weight_updates[1].abs().sum().item()  # L1 norm for input to hidden
            energy_consumption_hidden_to_output += weight_updates[2].abs().sum().item() + weight_updates[3].abs().sum().item()  # L1 norm for hidden to output
        cumulative_energy_consumption += energy_consumption
        cumulative_energy_consumption_input_to_hidden += energy_consumption_input_to_hidden
        cumulative_energy_consumption_hidden_to_output += energy_consumption_hidden_to_output
        energy_consumptions.append(cumulative_energy_consumption)
        energy_consumptions_input_to_hidden.append(cumulative_energy_consumption_input_to_hidden)
        energy_consumptions_hidden_to_output.append(cumulative_energy_consumption_hidden_to_output)
    final_energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption')
    plt.semilogy(accuracies, energy_consumptions_input_to_hidden, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption (Input to Hidden)')
    plt.semilogy(accuracies, energy_consumptions_hidden_to_output, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption (Hidden to Output)')
    plt.semilogy(range(1, len(accuracies) + 1), energy_consumptions, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption')
    plt.title('Cumulative Energy Consumption vs Epoch')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_energy_vs_epoch.png")
plt.semilogy(learning_rates, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Learning Rate')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_vs_energy.png")
plt.scatter(final_epochs, final_energy_consumptions, color='green', s=10)
                 (final_epochs[i], final_energy_consumptions[i]),
plt.ylabel('Cumulative Energy Consumption')
plt.title('Final Epoch and Energy Consumption for different Learning Rates')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/lr_vs_epoch_energy_scatter.png")
plt.scatter(final_epochs, final_energy_consumptions, color='green', s=10)
                 (final_epochs[i], final_energy_consumptions[i]),
padding_y = 0.1 * (max(final_energy_consumptions) - min(final_energy_consumptions))
plt.ylim(min(final_energy_consumptions) / 1.1, max(final_energy_consumptions) * 1.2)  # For log scale, use division and multiplication
plt.ylabel('Cumulative Energy Consumption')
plt.title('Final Epoch and Energy Consumption for different Learning Rates')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/lr_vs_epoch_energy_scatter.png")
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumption_input_to_hidden = 0
    cumulative_energy_consumption_hidden_to_output = 0
    energy_consumptions_input_to_hidden = []
    energy_consumptions_hidden_to_output = []
        energy_consumption = 0
        energy_consumption_input_to_hidden = 0 
        energy_consumption_hidden_to_output = 0 
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
            energy_consumption_input_to_hidden += weight_updates[0].abs().sum().item() + weight_updates[1].abs().sum().item()  # L1 norm for input to hidden
            energy_consumption_hidden_to_output += weight_updates[2].abs().sum().item() + weight_updates[3].abs().sum().item()  # L1 norm for hidden to output
        cumulative_energy_consumption += energy_consumption
        cumulative_energy_consumption_input_to_hidden += energy_consumption_input_to_hidden
        cumulative_energy_consumption_hidden_to_output += energy_consumption_hidden_to_output
        energy_consumptions.append(cumulative_energy_consumption)
        energy_consumptions_input_to_hidden.append(cumulative_energy_consumption_input_to_hidden)
        energy_consumptions_hidden_to_output.append(cumulative_energy_consumption_hidden_to_output)
    final_energy_consumptions.append(cumulative_energy_consumption)       
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim={hidden_dim}, num_layers=1')
    plt.ylabel('Cumulative Energy Consumption')
    plt.semilogy(accuracies, energy_consumptions_input_to_hidden, label=f'hidden_dim={hidden_dim}, num_layers=1')
    plt.ylabel('Cumulative Energy Consumption (Input to Hidden)')
    plt.semilogy(accuracies, energy_consumptions_hidden_to_output, label=f'hidden_dim={hidden_dim}, num_layers=1')
    plt.ylabel('Cumulative Energy Consumption (Hidden to Output)')
plt.semilogy(width_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Width')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/width_vs_energy.png")
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
        cumulative_energy_consumptions.append(cumulative_energy_consumption)
    final_energy_consumptions.append(cumulative_energy_consumption) 
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim=100, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')
plt.semilogy(depth_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Depth')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/depth_vs_energy.png")
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
        cumulative_energy_consumptions.append(cumulative_energy_consumption)
    final_energy_consumptions.append(cumulative_energy_consumption) 
    plt.plot(accuracies, energy_consumptions, label=f'hidden_dim=100, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')
    yticks = [10**i for i in range(int(np.log10(min(energy_consumptions))), int(np.log10(max(energy_consumptions)))+1)]
plt.plot(depth_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Depth')
yticks = [10**i for i in range(int(np.log10(min(final_energy_consumptions))), int(np.log10(max(final_energy_consumptions)))+1)]
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim={hidden_dim}, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'width=(64,128,256,512), depth={num_blocks}')
    plt.ylabel('Cumulative Energy Consumption')
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    final_energy_consumptions.append(cumulative_energy_consumption) 
    plt.semilogy(accuracies, energy_consumptions, label=f'width=(64,128,256,512), depth={num_blocks}')
    plt.ylabel('Cumulative Energy Consumption')
plt.semilogy(depth_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Depth')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESdepth_vs_energy.png")
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'width={width_mult}, num_blocks={depth}')
    plt.ylabel('Cumulative Energy Consumption')
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumption_input_to_hidden = 0
    cumulative_energy_consumption_hidden_to_output = 0
    energy_consumptions_input_to_hidden = []
    energy_consumptions_hidden_to_output = []
        energy_consumption = 0
        energy_consumption_input_to_hidden = 0 
        energy_consumption_hidden_to_output = 0 
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
            energy_consumption_input_to_hidden += weight_updates[0].abs().sum().item() + weight_updates[1].abs().sum().item()  # L1 norm for input to hidden
            energy_consumption_hidden_to_output += weight_updates[2].abs().sum().item() + weight_updates[3].abs().sum().item()  # L1 norm for hidden to output
        cumulative_energy_consumption += energy_consumption
        cumulative_energy_consumption_input_to_hidden += energy_consumption_input_to_hidden
        cumulative_energy_consumption_hidden_to_output += energy_consumption_hidden_to_output
        energy_consumptions.append(cumulative_energy_consumption)
        energy_consumptions_input_to_hidden.append(cumulative_energy_consumption_input_to_hidden)
        energy_consumptions_hidden_to_output.append(cumulative_energy_consumption_hidden_to_output)
    final_energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption')
    plt.semilogy(accuracies, energy_consumptions_input_to_hidden, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption (Input to Hidden)')
    plt.semilogy(accuracies, energy_consumptions_hidden_to_output, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption (Hidden to Output)')
    plt.semilogy(range(1, len(accuracies) + 1), energy_consumptions, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption')
    plt.title('Cumulative Energy Consumption vs Epoch')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_energy_vs_epoch.png")
plt.semilogy(learning_rates, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Learning Rate')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/learningrate_vs_energy.png")
plt.scatter(final_epochs, final_energy_consumptions, color='green', s=10)
                 (final_epochs[i], final_energy_consumptions[i]),
plt.ylabel('Cumulative Energy Consumption')
plt.title('Final Epoch and Energy Consumption for different Learning Rates')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/lr_vs_epoch_energy_scatter.png")
plt.scatter(final_epochs, final_energy_consumptions, color='green', s=10)
                 (final_epochs[i], final_energy_consumptions[i]),
padding_y = 0.1 * (max(final_energy_consumptions) - min(final_energy_consumptions))
plt.ylim(min(final_energy_consumptions) / 1.1, max(final_energy_consumptions) * 1.2)  # For log scale, use division and multiplication
plt.ylabel('Cumulative Energy Consumption')
plt.title('Final Epoch and Energy Consumption for different Learning Rates')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/lr_vs_epoch_energy_scatter.png")
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumption_input_to_hidden = 0
    cumulative_energy_consumption_hidden_to_output = 0
    energy_consumptions_input_to_hidden = []
    energy_consumptions_hidden_to_output = []
        energy_consumption = 0
        energy_consumption_input_to_hidden = 0 
        energy_consumption_hidden_to_output = 0 
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
            energy_consumption_input_to_hidden += weight_updates[0].abs().sum().item() + weight_updates[1].abs().sum().item()  # L1 norm for input to hidden
            energy_consumption_hidden_to_output += weight_updates[2].abs().sum().item() + weight_updates[3].abs().sum().item()  # L1 norm for hidden to output
        cumulative_energy_consumption += energy_consumption
        cumulative_energy_consumption_input_to_hidden += energy_consumption_input_to_hidden
        cumulative_energy_consumption_hidden_to_output += energy_consumption_hidden_to_output
        energy_consumptions.append(cumulative_energy_consumption)
        energy_consumptions_input_to_hidden.append(cumulative_energy_consumption_input_to_hidden)
        energy_consumptions_hidden_to_output.append(cumulative_energy_consumption_hidden_to_output)
    final_energy_consumptions.append(cumulative_energy_consumption)       
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim={hidden_dim}, num_layers=1')
    plt.ylabel('Cumulative Energy Consumption')
    plt.semilogy(accuracies, energy_consumptions_input_to_hidden, label=f'hidden_dim={hidden_dim}, num_layers=1')
    plt.ylabel('Cumulative Energy Consumption (Input to Hidden)')
    plt.semilogy(accuracies, energy_consumptions_hidden_to_output, label=f'hidden_dim={hidden_dim}, num_layers=1')
    plt.ylabel('Cumulative Energy Consumption (Hidden to Output)')
plt.semilogy(width_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Width')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/width_vs_energy.png")
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
        cumulative_energy_consumptions.append(cumulative_energy_consumption)
    final_energy_consumptions.append(cumulative_energy_consumption) 
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim=100, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')
plt.semilogy(depth_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Depth')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/depth_vs_energy.png")
plt.semilogy(depth_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Depth')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/depth_vs_energy.png")
print(final_energy_consumptions)
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
    cumulative_energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
        cumulative_energy_consumptions.append(cumulative_energy_consumption)
    final_energy_consumptions.append(cumulative_energy_consumption) 
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim=100, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')
plt.semilogy(depth_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Depth')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/depth_vs_energy.png")
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim={hidden_dim}, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim={hidden_dim}, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption with weight pruning')
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'width=(64,128,256,512), depth={num_blocks}')
    plt.ylabel('Cumulative Energy Consumption')
final_energy_consumptions = []
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    final_energy_consumptions.append(cumulative_energy_consumption) 
    plt.semilogy(accuracies, energy_consumptions, label=f'width=(64,128,256,512), depth={num_blocks}')
    plt.ylabel('Cumulative Energy Consumption')
plt.semilogy(depth_variations, final_energy_consumptions)
plt.ylabel('Cumulative Energy Consumption')
plt.title('Cumulative Energy Consumption vs Depth')
plt.savefig("/Users/tim/Desktop/postgraduate/research project/STUDY/diagram/RESdepth_vs_energy.png")
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'width={width_mult}, num_blocks={depth}')
    plt.ylabel('Cumulative Energy Consumption')
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'lr={lr}')
    plt.ylabel('Cumulative Energy Consumption')
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim={hidden_dim}, num_layers=1')
    plt.ylabel('Cumulative Energy Consumption')
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim=100, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')
    cumulative_energy_consumption = 0
    energy_consumptions = []
        energy_consumption = 0
            # Estimate the energy consumption as the sum of the weight update sizes
            energy_consumption += sum(weight_update_sizes)
        cumulative_energy_consumption += energy_consumption
        energy_consumptions.append(cumulative_energy_consumption)
    plt.semilogy(accuracies, energy_consumptions, label=f'hidden_dim={hidden_dim}, num_layers={num_layers}')
    plt.ylabel('Cumulative Energy Consumption')