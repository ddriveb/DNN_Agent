# Training Configuration

* Protocol: `DEEPRMSA_SOURCE_SEMANTIC_PAPER_PRIMARY_K50`
* Implementation: DeepRMSA source-semantic unmasked PyTorch port
* Algorithm: **single-process A2C (not A3C)**
* Training seeds: [42, 123, 456]
* Validation seeds: [42001, 42002, 42003]
* Epochs: 80
* Train warm-up per epoch: 1000
* Train requests per epoch: 5000
* Total training requests: 480000
* Validation requests: 3000

## Network
* Layers: 5 x 128 ELU
* Policy head normalized-columns std: 0.01
* Value head normalized-columns std: 1.0

## Hyperparameters
* gamma: 0.95
* entropy coefficient: 0.01
* value-loss coefficient: 1.0
* gradient clipping: 40.0
* learning rate: 1e-05
