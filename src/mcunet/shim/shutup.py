__all__ = ["silence"]


def silence() -> None:
    """
    Silences Tensorflow and Keras.
    """
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow("ERROR")

    import keras
    keras.utils.disable_interactive_logging()
