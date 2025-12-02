"""Recommendation ML models"""
from .neural_collaborative import NeuralCollaborativeFiltering
from .neural_content_based import NeuralContentBased

__all__ = ["NeuralCollaborativeFiltering", "NeuralContentBased"]

