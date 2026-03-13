"""
AutoMixAI – ANN Model Definition

Builds a dense (fully-connected) neural network for binary beat
classification: each audio frame is labelled as *beat* (1) or
*non-beat* (0).
"""

from tensorflow import keras
from tensorflow.keras import layers

from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_model(input_dim: int) -> keras.Model:
    """
    Construct and compile the beat-detection ANN.

    Architecture::

        Input(input_dim)
          → Dense(128, ReLU) → Dropout(0.3)
          → Dense(64,  ReLU) → Dropout(0.2)
          → Dense(1, Sigmoid)

    Args:
        input_dim: Number of features per frame (e.g. ``n_mfcc + 3``).

    Returns:
        A compiled Keras ``Model`` ready for training.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    logger.info("ANN model built — input_dim=%d", input_dim)
    return model
