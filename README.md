# VeriDP: Verifiable Differentially Private Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/LibTorch-2.x-EE4C2C.svg" alt="LibTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A cryptographic framework for **verifiable differentially private stochastic gradient descent (DP-SGD)**. VeriDP generates zero-knowledge proofs that the training process correctly applied differential privacy mechanisms, without revealing the training data or model internals.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Verifiable DP-SGD** | Cryptographic proofs for gradient clipping, noise addition, and weight updates |
| **GKR Protocol** | Sub-linear verification using the GKR interactive proof system |
| **IVC Aggregation** | Incrementally verifiable computation for efficient proof composition |
| **Box-Muller Proofs** | GKR circuits for verifiable Gaussian noise generation |
| **Dataset Commitment** | Merkle tree commitment ensuring training on declared dataset |
| **Privacy Accountant** | Renyi DP composition tracking with (epsilon, delta) guarantees |
| **Performance Metrics** | Detailed timing breakdown for prover/verifier operations |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VeriDP Training                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Dataset    │  │   DP-SGD     │  │   Proof Generation   │   │
│  │  Commitment  │──│   Training   │──│   (GKR + Sumcheck)   │   │
│  │ (Merkle Tree)│  │  (LibTorch)  │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│         │                  │                    │               │
│         ▼                  ▼                    ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Membership  │  │   Privacy    │  │   IVC Aggregation    │   │
│  │    Proofs    │  │  Accountant  │  │   (Fiat-Shamir)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Verifier     │
                    │  (Sub-linear)   │
                    └─────────────────┘
```

---

## Prerequisites

- **C++17** compatible compiler (GCC 9+, Clang 10+)
- **CMake** 3.10+
- **LibTorch** 2.x (PyTorch C++ API)
- **OpenMP** for parallelization
- **Optional**: MCL library for 256-bit BLS12-381 field

### Installing LibTorch

```bash
# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# Or GPU version (CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
```

---

## Quick Start

### Build

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/veriDP-997D/
cd veridp

# Create build directory
mkdir build && cd build

# Configure (point to your LibTorch installation)
cmake -DTORCH_ROOT=/path/to/libtorch ..

# Build
make train_veridp -j$(nproc)
```

### Download MNIST

```bash
mkdir -p data && cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

### Run

```bash
# Basic run (2 batches)
./train_veridp ../data --max-batches=2

# Full training with custom parameters
./train_veridp ../data --max-batches=10

# Use simple MLP with forward/backward proofs
./train_veridp ../data --max-batches=2 --simple-mlp

# CIFAR-10 with ResNet18
./train_veridp ../data --dataset=cifar10 --model=resnet18 --max-batches=2

# Synthetic linear regression (no dataset files required)
./train_veridp --dataset=linear --model=linear --linear-size=100 --linear-dim=16 --batch=5 --max-steps=20 --lr=0.1 --sigma=1.0

# Synthetic logistic regression (no dataset files required)
./train_veridp --dataset=logistic --model=logistic --logistic-size=100 --logistic-dim=16 --batch=5 --max-steps=20 --lr=0.1 --sigma=1.0
```

### Benchmark Scripts

```bash
# MNIST benchmark suite
bash run_benchmark.sh
bash run_parameter_sweep.sh
bash run_full_benchmark.sh

# CIFAR-10 + ResNet benchmark suite
bash run_benchmark_cifar_resnet.sh
bash run_parameter_sweep_cifar_resnet.sh
bash run_full_benchmark_cifar_resnet.sh

# Synthetic linear/logistic grid benchmark
bash run_linear_logistic_grid.sh
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_norm` | 10.0 | Gradient clipping threshold (C) - tune based on model |
| `learning_rate` | 0.01 | SGD learning rate (eta) |
| `noise_multiplier` | 1.1 | Noise scale (sigma) for DP guarantee |
| `batch_size` | 32 | Samples per batch |
| `epochs` | 1 | Training epochs |
| `max_batches` | -1 | Limit batches per epoch (-1 = unlimited) |
| `max_steps` | -1 | Global limit on total minibatch updates across epochs |
| `use_simple_mlp` | false | Use MLP with full forward/backward proofs |
| `dataset` | `mnist` | Dataset selector: `mnist`, `cifar10`, `linear`, `logistic` |
| `model` | dataset default | Model selector: `mnist_cnn`, `cifar_cnn`, `resnet18`, `simple_mlp`, `linear`, `logistic` |
| `linear_size` | 1000 | Synthetic linear dataset size |
| `linear_dim` | 5 | Synthetic linear feature dimension |
| `linear_noise_range` | 10 | Synthetic linear noise range (`0` disables dataset-generation noise) |
| `linear_noise_increment` | 0.001 | Synthetic linear noise increment |
| `logistic_size` | 1000 | Synthetic logistic dataset size |
| `logistic_dim` | 5 | Synthetic logistic feature dimension |
| `logistic_noise_range` | 10 | Synthetic logistic noise range (`0` disables dataset-generation noise) |
| `logistic_noise_increment` | 0.001 | Synthetic logistic noise increment |

---

## Output Example

```
=== VeriDP: Verifiable DP-SGD Training ===
Configuration:
  Clip norm: 1
  Learning rate: 0.01
  Noise multiplier: 1.1
  Batch size: 32
  Epochs: 1
  Max batches: 2

[Dataset] Committing 64 samples to Merkle tree...
[Dataset] Commitment complete. Root hash computed.

[Epoch 1/1] Batch 1/2: 32 samples, 4 proofs
[Epoch 1/1] Batch 2/2: 32 samples, 4 proofs

=== VERIFICATION ===
[Verification] Proof size: 12.453 KB
[Verification] All structure checks passed
[Verification] Verification completed in 2.34 ms

╔══════════════════════════════════════════════════════════════╗
║              VeriDP PERFORMANCE METRICS                      ║
╠══════════════════════════════════════════════════════════════╣
║ GENERAL                                                      ║
║   Batches processed:               2                         ║
║   Samples processed:              64                         ║
║   Total proofs generated:          8                         ║
╠══════════════════════════════════════════════════════════════╣
║ PROOF SIZE                                                   ║
║   Total proof size:          12.453 KB                       ║
║   Avg per batch:              6.227 KB                       ║
╠══════════════════════════════════════════════════════════════╣
║ PROVER TIME                                                  ║
║   Total prover time:         156.23 ms                       ║
║   ├─ Clipping proof:          12.45 ms ( 7.97%)              ║
║   ├─ Noise generation:        89.12 ms (57.04%)              ║
║   ├─ Noise addition:          23.45 ms (15.01%)              ║
║   └─ Weight update:           31.21 ms (19.98%)              ║
╠══════════════════════════════════════════════════════════════╣
║ VERIFIER TIME                                                ║
║   Total verifier time:         2.34 ms                       ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║              PRIVACY BUDGET SUMMARY                          ║
╠══════════════════════════════════════════════════════════════╣
║ Noise multiplier (sigma):            1.100000                    ║
║ Sampling rate (q):               0.001067                    ║
║ Target delta:                        0.000010                    ║
║ Total iterations:                     2                      ║
╠══════════════════════════════════════════════════════════════╣
║ Final epsilon:                         0.4523                      ║
║ Final delta:                         0.0000                      ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Proof System Details

### What is Proved

1. **Gradient Clipping**: Per-sample gradient norms are bounded by C
2. **Gradient Averaging**: Clipped gradients are correctly averaged
3. **Noise Generation**: Gaussian noise via Box-Muller transform is valid
4. **Noise Addition**: Noise is correctly added to averaged gradients
5. **Weight Update**: Weights are updated as `w_new = w_old - eta * noisy_grad`
6. **Dataset Membership**: Each batch sample belongs to the committed dataset

### Cryptographic Primitives

| Component | Implementation |
|-----------|----------------|
| Hash Function | SHA-256 (SHA-NI accelerated) |
| Field | 61-bit Mersenne prime (or 256-bit BLS12-381) |
| Commitment | Merkle Tree (dataset), Polynomial (proofs) |
| Interactive Proof | GKR Protocol with Fiat-Shamir transform |
| Aggregation | Incremental Verifiable Computation (IVC) |

---

## Project Structure

```
veridp/
├── train_veridp.cpp        # Main training with proofs
├── dp_sgd_libtorch.h       # DP-SGD implementation
├── veridp_proofs.h         # Proof generation for DP-SGD steps
├── veridp_utils.h          # Tensor <-> Field conversions
├── veridp_metrics.h        # Performance tracking
├── veridp_forward_backward.h # MLP forward/backward proofs
├── privacy_accountant.h    # RDP composition tracker
├── dataset_commitment.h/cpp # Merkle tree commitment
├── box_muller_circuits.h   # GKR circuits for Box-Muller
├── mnist_cnn.h             # CNN model
├── simple_mlp.h            # Simple MLP model
├── Summer code/            # Cryptographic infrastructure
│   ├── fieldElement*.hpp/cpp  # Field arithmetic
│   ├── GKR.cpp/h           # GKR protocol
│   ├── verifier.cpp        # Proof verification
│   ├── ivc_adapter.cpp     # IVC aggregation
│   ├── merkle_tree.cpp     # Merkle tree operations
│   ├── mimc.cpp            # MIMC hash
│   └── ...
├── CMakeLists.txt          # Build configuration
└── data/                   # MNIST dataset
```

---

Additional model/dataset modules in the updated repo:
- `cifar_cnn.h`, `resnet18.h`
- `linear_regression.h`, `logistic_regression.h`
- `synthetic_linear_dataset.h`, `synthetic_logistic_dataset.h`
- `run_benchmark_cifar_resnet.sh`, `run_parameter_sweep_cifar_resnet.sh`, `run_full_benchmark_cifar_resnet.sh`
- `run_linear_logistic_grid.sh`

## Advanced Configuration

### 256-bit Field (BLS12-381)

For higher security with the BLS12-381 scalar field:

```bash
# Install MCL library
git clone https://github.com/herumi/mcl.git
cd mcl && make -j$(nproc)
cd ..

# Build with 256-bit field
cmake -DUSE_FIELD_256=ON -DTORCH_ROOT=/path/to/libtorch ..
make train_veridp -j$(nproc)
```

---

## References

- **DP-SGD**: Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
- **GKR Protocol**: Goldwasser, Kalai, Rothblum, "Delegating Computation" (STOC 2008)
- **Virgo**: Zhang et al., "Transparent Polynomial Delegation" (IEEE S&P 2020)
- **RDP**: Mironov, "Renyi Differential Privacy" (CSF 2017)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a PR.

---

<p align="center">
  <b>VeriDP</b> - Trustworthy Machine Learning through Verifiable Privacy
</p>

