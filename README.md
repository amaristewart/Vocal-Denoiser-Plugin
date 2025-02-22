# Vocal Denoiser Plugin

## Overview
The **Vocal Denoiser Plugin** is an advanced audio processing tool designed to enhance speech clarity by effectively reducing background noise in real time. This plugin utilizes a convolutional neural network (CNN) to suppress unwanted noise while preserving speech integrity.

## Features
- **Real-Time Noise Suppression**: Uses a pre-trained deep learning model to analyze and remove noise from speech.
- **Short-Time Fourier Transform (STFT)**: Converts the time-domain signal into a frequency-domain representation for precise processing.
- **Voice Activity Detection (VAD)**: Identifies and isolates speech from non-speech regions, ensuring that the noise suppression is applied only where necessary.
- **Adaptive Noise Gate**: Dynamically adjusts threshold levels based on detected speech probability to further refine noise reduction.
- **Multiple Noise Profiles**: Users can select from predefined noise profiles:
  - Washing Machine
  - White Noise
  - Crowd Noise
- **Sample Rate Conversion**: Supports multiple sample rates and automatically converts signals to ensure compatibility with various audio workflows.
- **User Customization**: Adjustable *Strength* parameter allows fine-tuning of the denoising effect.

## Implementation
The plugin processes incoming audio through the following stages:
1. **Noise Profile Selection**: Users select a noise type that best matches their environment.
2. **Preprocessing with STFT**: The signal is transformed into its frequency representation using Short-Time Fourier Transform (STFT).
3. **Neural Network Denoising**: A CNN trained on noisy and clean speech pairs processes the STFT data, estimating a clean speech output.
4. **Blending Mechanism**: The denoised output is blended with the original signal based on user-defined strength levels to maintain natural speech quality.
5. **VAD Integration**: The VAD model detects speech segments, ensuring that the noise suppression is applied only when necessary.
6. **Final Reconstruction**: The processed signal is converted back to the time domain using the Inverse Short-Time Fourier Transform (ISTFT) and outputted at the original sample rate.

