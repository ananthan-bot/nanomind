"""
nanomind/merge/soup.py — Model Soups: Uniform Soup and Greedy Soup (Wortsman et al., ICML 2022).
Averages fine-tuned checkpoints for free performance and out-of-distribution robustness.
"""
from typing import List, Dict, Any, Callable, Optional
import copy
import torch
import torch.nn as nn


class ModelSoup:
    """
    Constructs model soups by averaging model weights.
    """

    @staticmethod
    def uniform_soup(models: List[nn.Module]) -> nn.Module:
        """
        Uniformly averages all model weights: theta = 1/N * sum theta_i.
        """
        if not models:
            raise ValueError("No models provided for soup")

        soup = copy.deepcopy(models[0])
        soup_params = dict(soup.named_parameters())
        n = len(models)

        # Initialize to 0
        for p in soup_params.values():
            p.data.zero_()

        # Accumulate weights
        for model in models:
            for name, param in model.named_parameters():
                if name in soup_params:
                    soup_params[name].data.add_(param.data, alpha=1.0 / n)

        return soup

    @staticmethod
    def greedy_soup(
        models: List[nn.Module],
        eval_fn: Callable[[nn.Module], float],
    ) -> nn.Module:
        """
        Greedy soup: iteratively adds models that improve evaluation score.
        eval_fn: takes model and returns metric (higher is better).
        """
        if not models:
            raise ValueError("No models provided")

        # Rank models by individual performance
        scored = [(m, eval_fn(m)) for m in models]
        scored.sort(key=lambda x: x[1], reverse=True)

        current_soup = copy.deepcopy(scored[0][0])
        best_score = scored[0][1]
        active_models = [scored[0][0]]

        for model, _ in scored[1:]:
            candidate_models = active_models + [model]
            candidate_soup = ModelSoup.uniform_soup(candidate_models)
            score = eval_fn(candidate_soup)

            if score > best_score:
                best_score = score
                active_models = candidate_models
                current_soup = candidate_soup

        return current_soup
