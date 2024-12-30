# NVIDIA NeMo

NVIDIA NeMo is an open-source toolkit for building, training, and deploying conversational AI models. It provides a rich set of pre-built modules for ASR (Automatic Speech Recognition), NLP (Natural Language Processing), and TTS (Text-to-Speech) tasks.

## How it works

NeMo is built on top of PyTorch and uses a modular approach to building neural networks. It provides:

1. Collections: Pre-built neural network architectures for ASR, NLP, and TTS.
2. Neural Modules: Building blocks that represent data layers, encoders, decoders, language models, loss functions, etc.
3. Models: High-level abstractions that string together Neural Modules to create full neural network graphs.

NeMo makes it easy to compose complex neural architectures using pre-built modules, allowing researchers and developers to quickly experiment with different model configurations.

## Key Features

- Pre-trained models for ASR, NLP, and TTS tasks
- Easy-to-use API for training and fine-tuning models
- Distributed training support
- Mixed precision training
- Deployment-ready with NVIDIA Triton Inference Server

For more information and the full source code, visit the [NVIDIA NeMo GitHub repository](https://github.com/NVIDIA/NeMo).

