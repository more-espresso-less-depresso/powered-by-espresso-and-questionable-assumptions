import matplotlib.pyplot as plt

def visualize_sample(sample, activations, num_layers, dataset_name=None):
    fig, axs = plt.subplots(1,1, figsize=(10, 6))
    axs.imshow(activations, aspect='auto', cmap='viridis')
    axs.set(title=f'Prompt: {sample}', xlabel='Tokens', ylabel='Layers')
    axs.set_yticks(ticks=list(range(num_layers)), labels=[f'{i+1}' for i in reversed(range(num_layers))])
    fig.colorbar(axs.images[0], label='Average Activation')
    plt.tight_layout()
    if dataset_name:
        plt.savefig(f'{dataset_name}_single_sample_activations.png')
        plt.close()
    else:
        plt.show()
    
def visualize_activations(activations, name, num_layers, dataset_name=None):
    fig, axs = plt.subplots(1,1, figsize=(10, 10))
    axs.imshow(activations, aspect='auto', cmap='viridis')
    axs.set(title=name, xlabel='Tokens', ylabel='Layers')
    axs.set_yticks(ticks=list(range(num_layers)), labels=[f'{i+1}' for i in reversed(range(num_layers))])
    fig.colorbar(axs.images[0], label='Average Activation')
    plt.tight_layout()
    
    if dataset_name:
        plt.savefig(f'{dataset_name}_{name}_activations.png')
        plt.close()
    else:
        plt.show()
    
def visualize_comparison_activations(benign_token_activations, harmful_token_activations, activation_type, num_layers, dataset_name=None):
    vmin = min(benign_token_activations.min(), harmful_token_activations.min())
    vmax = max(benign_token_activations.max(), harmful_token_activations.max())

    fig, axs = plt.subplots(1, 2, figsize=(20, 5), sharey=True, gridspec_kw={'wspace':0.05}, dpi=300)

    im0 = axs[0].imshow(benign_token_activations, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set(title=f'Benign {activation_type}', xlabel='Tokens', ylabel='Layers')
    axs[0].set_yticks([0])
    axs[0].set_yticklabels([f'{num_layers}'])

    im1 = axs[1].imshow(harmful_token_activations, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set(title=f'Harmful {activation_type}', xlabel='Tokens')

    # Create a single colorbar on the right side of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im0, cax=cbar_ax, label='Average Activation')

    if dataset_name:
        plt.savefig(f'{dataset_name}_{activation_type}_comparison_activations.png', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
        
def visualize_triple_activations(safe_activations, unsafe_activations, refusal_activations, activation_type, num_layers, dataset_name=None):
    vmin = min(safe_activations.min(), unsafe_activations.min(), refusal_activations.min())
    vmax = max(safe_activations.max(), unsafe_activations.max(), refusal_activations.max())

    fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True, gridspec_kw={'wspace':0.05}, dpi=300)

    im0 = axs[0].imshow(safe_activations, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set(title=f'Benign {activation_type}', xlabel='Tokens', ylabel='Layers')
    axs[0].set_yticks([0])
    axs[0].set_yticklabels([f'{num_layers}'])

    im1 = axs[1].imshow(unsafe_activations, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set(title=f'Harmful {activation_type}', xlabel='Tokens')

    im2 = axs[2].imshow(refusal_activations, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].set(title=f'Refusal {activation_type}', xlabel='Tokens')

    # Create a single colorbar on the right side of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im0, cax=cbar_ax, label='Average Activation')

    #plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar

    if dataset_name:
        plt.savefig(f'{dataset_name}_{activation_type}_triple_activations.png', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()