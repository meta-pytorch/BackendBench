#!/usr/bin/env python3
"""
Sample Input Generator for HuggingFace Model Testing

This module provides configurable functions to generate sample inputs for different
types of models. Developers can easily modify the input sizes, content, and other
parameters by editing the configuration variables or functions in this file.
"""

import math
import sys
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

# =============================================================================
# CONFIGURATION SECTION - Modify these to change input characteristics
# =============================================================================

# Text input configuration
# read from frankenstein.txt
LONG_TEXT_INPUT = open("frankenstein.txt", "r").read()
DEFAULT_TEXT_INPUT = open("pytorch_readme.txt", "r").read()
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQUENCE_LENGTH = 512

# Image input configuration
DEFAULT_IMAGE_SIZE = (224, 224)  # (width, height)
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_IMAGE_BATCH_SIZE = 1

# Audio input configuration
DEFAULT_AUDIO_SAMPLE_RATE = 3000
DEFAULT_AUDIO_DURATION_SECONDS = 1.0
DEFAULT_AUDIO_CHANNELS = 80

# Time series configuration
DEFAULT_TIME_SERIES_LENGTH = 100
DEFAULT_TIME_SERIES_FEATURES = 1

# Vision tensor configuration
DEFAULT_VISION_TENSOR_SHAPE = (1, 3, 224, 224)  # (batch, channels, height, width)

# Text tensor configuration
DEFAULT_TEXT_VOCAB_SIZE = 32128  # T5 default vocab size
DEFAULT_TOKEN_SEQUENCE_LENGTH = 50

# Classification labels
DEFAULT_CLASSIFICATION_LABELS = ["positive", "negative", "neutral"]
DEFAULT_OBJECT_DETECTION_LABELS = ["cat", "dog", "person", "car", "building"]
DEFAULT_CLIP_TEXT_LABELS = ["a photo of a cat", "a photo of a dog"]

# Protein sequences
DEFAULT_PROTEIN_SEQUENCE_LONG = (
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
)
DEFAULT_PROTEIN_SEQUENCE_SHORT = "MKTVRQER"

# Sentence transformer test sentences
DEFAULT_SENTENCE_TRANSFORMER_TEXTS = [
    "Hello, this is a test input for the model.",
    "Another test sentence",
]

DEFAULT_SENTENCE_TRANSFORMER_ENCODING_TEXTS = [
    "Hello, this is a test input for the model.",
    "Another test sentence for encoding",
]

# =============================================================================
# BASIC INPUT GENERATION FUNCTIONS
# =============================================================================


def create_text_input(text: str = None) -> str:
    """Create dummy text input for text-based models."""
    return text if text is not None else DEFAULT_TEXT_INPUT


def create_image_input(
    size: Tuple[int, int] = None, channels: int = None, random_seed: int = None
) -> Image.Image:
    """
    Create dummy image input for vision models.

    Args:
        size: Image size as (width, height)
        channels: Number of channels (3 for RGB, 1 for grayscale)
        random_seed: Random seed for reproducibility
    """
    if size is None:
        size = DEFAULT_IMAGE_SIZE
    if channels is None:
        channels = DEFAULT_IMAGE_CHANNELS

    if random_seed is not None:
        np.random.seed(random_seed)

    if channels == 1:
        # Grayscale
        image_array = np.random.randint(0, 256, (*size[::-1], 1), dtype=np.uint8)
        image_array = np.squeeze(image_array, axis=2)
        return Image.fromarray(image_array, mode="L")
    else:
        # RGB
        image_array = np.random.randint(0, 256, (*size[::-1], channels), dtype=np.uint8)
        return Image.fromarray(image_array)


def create_audio_input(
    sample_rate: int = None,
    duration_seconds: float = None,
    channels: int = None,
    random_seed: int = None,
) -> np.ndarray:
    """
    Create dummy audio input for audio models.

    Args:
        sample_rate: Audio sample rate in Hz
        duration_seconds: Duration of audio in seconds
        channels: Number of audio channels
        random_seed: Random seed for reproducibility
    """
    if sample_rate is None:
        sample_rate = DEFAULT_AUDIO_SAMPLE_RATE
    if duration_seconds is None:
        duration_seconds = DEFAULT_AUDIO_DURATION_SECONDS
    if channels is None:
        channels = DEFAULT_AUDIO_CHANNELS

    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = int(sample_rate * duration_seconds)

    if channels == 1:
        return np.random.randn(num_samples).astype(np.float32)
    else:
        return np.random.randn(num_samples, channels).astype(np.float32)


def create_time_series_input(
    length: int = None,
    features: int = None,
    add_trend: bool = True,
    add_seasonality: bool = True,
    random_seed: int = None,
) -> List[float]:
    """
    Create dummy time series input for forecasting models.

    Args:
        length: Number of time steps
        features: Number of features (for multivariate series)
        add_trend: Whether to add a trend component
        add_seasonality: Whether to add seasonality
        random_seed: Random seed for reproducibility
    """
    if length is None:
        length = DEFAULT_TIME_SERIES_LENGTH
    if features is None:
        features = DEFAULT_TIME_SERIES_FEATURES

    if random_seed is not None:
        np.random.seed(random_seed)

    time_series = []
    for i in range(length):
        value = 0.0

        # Add trend
        if add_trend:
            value += i * 0.1

        # Add seasonality
        if add_seasonality:
            value += 10 * math.sin(i * 0.1)

        # Add noise
        value += np.random.normal(0, 0.1)

        time_series.append(float(value))

    return time_series


# =============================================================================
# TENSOR GENERATION FUNCTIONS
# =============================================================================


def create_vision_tensor(
    shape: Tuple[int, ...] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    random_seed: int = None,
) -> torch.Tensor:
    """
    Create dummy vision tensor input.

    Args:
        shape: Tensor shape (batch, channels, height, width)
        dtype: Tensor data type
        device: Device to create tensor on
        random_seed: Random seed for reproducibility
    """
    if shape is None:
        shape = DEFAULT_VISION_TENSOR_SHAPE

    if random_seed is not None:
        torch.manual_seed(random_seed)

    return torch.randn(shape, dtype=dtype, device=device)


def create_audio_tensor(
    batch_size: int = None,
    num_samples: int = None,
    channels: int = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    random_seed: int = None,
) -> torch.Tensor:
    """
    Create dummy audio tensor input.

    Args:
        batch_size: Batch size
        num_samples: Number of audio samples
        channels: Number of audio channels (1 for mono, 2 for stereo)
        dtype: Tensor data type
        device: Device to create tensor on
        random_seed: Random seed for reproducibility
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if num_samples is None:
        num_samples = int(DEFAULT_AUDIO_SAMPLE_RATE * DEFAULT_AUDIO_DURATION_SECONDS)
    if channels is None:
        channels = DEFAULT_AUDIO_CHANNELS

    if random_seed is not None:
        torch.manual_seed(random_seed)

    if channels == 1:
        shape = (batch_size, num_samples)
    else:
        shape = (batch_size, channels, num_samples)

    return torch.randn(shape, dtype=dtype, device=device)


def create_text_token_tensor(
    batch_size: int = None,
    sequence_length: int = None,
    vocab_size: int = None,
    dtype: torch.dtype = torch.long,
    device: str = "cpu",
    random_seed: int = None,
) -> torch.Tensor:
    """
    Create dummy text token tensor input.

    Args:
        batch_size: Batch size
        sequence_length: Length of token sequence
        vocab_size: Vocabulary size for token IDs
        dtype: Tensor data type
        device: Device to create tensor on
        random_seed: Random seed for reproducibility
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if sequence_length is None:
        sequence_length = DEFAULT_TOKEN_SEQUENCE_LENGTH
    if vocab_size is None:
        vocab_size = DEFAULT_TEXT_VOCAB_SIZE

    if random_seed is not None:
        torch.manual_seed(random_seed)

    return torch.randint(
        0, vocab_size, (batch_size, sequence_length), dtype=dtype, device=device
    )


def create_mask_token_indices(
    sequence_length: int, num_masks: int = 1, random_seed: int = None
) -> torch.Tensor:
    """
    Create random indices for mask tokens.

    Args:
        sequence_length: Length of the sequence
        num_masks: Number of positions to mask
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    return torch.randint(0, sequence_length, (num_masks,))


def create_trimap_tensor(
    batch_size: int = None,
    height: int = 224,
    width: int = 224,
    device: str = "cpu",
    random_seed: int = None,
) -> torch.Tensor:
    """
    Create dummy trimap tensor for image matting models.

    Args:
        batch_size: Batch size
        height: Image height
        width: Image width
        device: Device to create tensor on
        random_seed: Random seed for reproducibility
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Create trimap with values 0 (background), 127 (unknown), 254 (foreground)
    return torch.randint(0, 3, (batch_size, height, width), device=device) * 127


# =============================================================================
# SPECIALIZED INPUT GENERATION FUNCTIONS
# =============================================================================


def create_protein_sequence(sequence_type: str = "long") -> str:
    """
    Create protein sequence for protein folding models.

    Args:
        sequence_type: "long" or "short" sequence
    """
    if sequence_type == "long":
        return DEFAULT_PROTEIN_SEQUENCE_LONG
    else:
        return DEFAULT_PROTEIN_SEQUENCE_SHORT


def create_classification_labels(
    label_type: str = "sentiment", custom_labels: List[str] = None
) -> List[str]:
    """
    Create classification labels for zero-shot classification.

    Args:
        label_type: Type of labels ("sentiment", "objects", "custom")
        custom_labels: Custom label list
    """
    if custom_labels is not None:
        return custom_labels

    if label_type == "sentiment":
        return DEFAULT_CLASSIFICATION_LABELS
    elif label_type == "objects":
        return DEFAULT_OBJECT_DETECTION_LABELS
    else:
        return DEFAULT_CLASSIFICATION_LABELS


def create_clip_text_labels(custom_labels: List[str] = None) -> List[str]:
    """
    Create text labels for CLIP models.

    Args:
        custom_labels: Custom label list
    """
    if custom_labels is not None:
        return custom_labels
    return DEFAULT_CLIP_TEXT_LABELS


def create_sentence_transformer_texts(
    text_type: str = "default", custom_texts: List[str] = None
) -> List[str]:
    """
    Create text lists for sentence transformer models.

    Args:
        text_type: "default" or "encoding"
        custom_texts: Custom text list
    """
    if custom_texts is not None:
        return custom_texts

    if text_type == "encoding":
        return DEFAULT_SENTENCE_TRANSFORMER_ENCODING_TEXTS
    else:
        return DEFAULT_SENTENCE_TRANSFORMER_TEXTS


# =============================================================================
# COMPOSITE INPUT GENERATION FUNCTIONS
# =============================================================================


def create_zero_shot_classification_input(
    text: str = None, candidate_labels: List[str] = None
) -> Dict[str, Union[str, List[str]]]:
    """Create input for zero-shot classification models."""
    return {
        "sequences": create_text_input(text),
        "candidate_labels": create_classification_labels(
            custom_labels=candidate_labels
        ),
    }


def create_zero_shot_image_classification_input(
    image_size: Tuple[int, int] = None, candidate_labels: List[str] = None
) -> Dict[str, Union[Image.Image, List[str]]]:
    """Create input for zero-shot image classification models."""
    return {
        "image": create_image_input(size=image_size),
        "candidate_labels": create_classification_labels("objects", candidate_labels),
    }


def create_clip_input(
    image_size: Tuple[int, int] = None, text_labels: List[str] = None
) -> Tuple[Image.Image, List[str]]:
    """Create input for CLIP models."""
    return (create_image_input(size=image_size), create_clip_text_labels(text_labels))


def create_time_series_tensor_variants(
    time_series_data: List[float] = None, include_batch_dim: bool = True
) -> List[torch.Tensor]:
    """
    Create different tensor format variants for time series data.

    Args:
        time_series_data: Raw time series data
        include_batch_dim: Whether to include batch dimension
    """
    if time_series_data is None:
        time_series_data = create_time_series_input()

    variants = []

    if include_batch_dim:
        # [batch, sequence]
        variants.append(torch.tensor([time_series_data]).float())
        # [batch, features, sequence]
        variants.append(torch.tensor([[time_series_data]]).float())

    # [sequence] - unsqueezed to [batch, sequence]
    variants.append(torch.tensor(time_series_data).float().unsqueeze(0))

    return variants


def create_chronos_input(
    batch_size: int = None,
    sequence_length: int = 20,
    vocab_size: int = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create input for Chronos time series models.

    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        vocab_size: Vocabulary size
        device: Device to create tensors on
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if vocab_size is None:
        vocab_size = DEFAULT_TEXT_VOCAB_SIZE

    input_ids = torch.randint(
        0, vocab_size, (batch_size, sequence_length), device=device
    )
    decoder_input_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)

    return {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_default_input_for_pipeline(pipeline_tag: str) -> Any:
    """
    Get appropriate default input for a given pipeline tag.

    Args:
        pipeline_tag: HuggingFace pipeline tag
    """
    if "image" in pipeline_tag or pipeline_tag == "object-detection":
        return create_image_input()
    elif "audio" in pipeline_tag:
        return create_audio_input()
    elif pipeline_tag == "zero-shot-classification":
        return create_zero_shot_classification_input()
    elif pipeline_tag == "zero-shot-image-classification":
        return create_zero_shot_image_classification_input()
    elif pipeline_tag == "automatic-speech-recognition":
        return create_speech_input()
    else:
        return create_text_input()


def create_speech_input():
    # example from https://huggingface.co/openai/whisper-large-v3
    dataset = load_dataset(
        "distil-whisper/librispeech_long", "clean", split="validation"
    )
    sample = dataset[0]["audio"]
    return {"inputs": sample, "return_timestamps": True}


def get_default_tensor_for_model_type(model_type: str) -> torch.Tensor:
    """
    Get appropriate default tensor for a given model type.

    Args:
        model_type: Model type string
    """
    if model_type == "text":
        return create_text_token_tensor()
    elif model_type in ["vision", "object-detection", "clip"]:
        return create_vision_tensor()
    elif model_type == "audio":
        return create_audio_tensor()
    else:
        # Default to vision-like input
        return create_vision_tensor()
