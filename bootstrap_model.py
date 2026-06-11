"""
Bootstrap script to create an untrained model.keras file.
Run this when you don't have a dataset yet but need the app to load.
"""

from train import build_cnn_model, save_untrained_model

if __name__ == "__main__":
    save_untrained_model()
