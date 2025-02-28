import os
from typing import Dict, List

import seaborn as sns
import matplotlib.pyplot as plt
import torch


class RetrievalEvaluator:
    def __init__(
            self,
            top_k: List[int],
            metrics: List[str],
            compute_per_class: bool = False,
        ):
        """Initialize retrieval evaluator with list of k values to compute recall@k.

        Args:
            top_k (list): List of k values for computing recall@k metrics
        """
        self.top_k = top_k
        self.compute_per_class = compute_per_class
        self.all_metrics = {
            'hits': self._compute_hits_at_k,
            'precision': self._compute_precision_at_k,
            'mrr': self._compute_mrr_at_k,
        }
        self.metrics = {
            metric: self.all_metrics[metric]
            for metric in metrics
        }

    def _compute_hits_at_k(self, correct: torch.Tensor):
        """Compute hits@k.

        Args:
            correct (torch.Tensor): Boolean tensor indicating correct predictions

        """
        return float(torch.sum(correct.any(dim=1)) / len(correct))

    def _compute_precision_at_k(self, correct: torch.Tensor):
        """
        Args:
            correct: Boolean tensor indicating correct predictions
            k (int): Number of top predictions to consider
        """
        return float(torch.mean(torch.sum(correct, dim=1).float() / correct.shape[1]))

    def _compute_mrr_at_k(self, correct: torch.Tensor):
        """
        Args:
            correct: Boolean tensor indicating correct predictions
        """
        ranks = torch.where(
            correct.any(dim=1),
            torch.argmax(correct.int(), dim=1) + 1,
            torch.tensor(float('inf')).to(correct.device)
        )
        return float(torch.mean(1.0 / ranks.float()))

    def evaluate(self, sorted_indices, query_labels):
        """Evaluate retrieval performance using similarity matrix and ground truth labels.

        Args:
            similarity_matrix: Matrix of shape [num_queries, num_candidates] containing similarity scores
            query_labels: Array of shape [num_queries] containing ground truth class labels for each query

        Returns:
            tuple: (metrics, metrics_per_class) containing overall and per-class metrics
        """
        if not isinstance(sorted_indices, torch.Tensor):
            sorted_indices = torch.tensor(sorted_indices)
        if not isinstance(query_labels, torch.Tensor):
            query_labels = torch.tensor(query_labels)


        metrics = {}
        metrics_per_class = {}

        for k in self.top_k:
            top_k_indices = sorted_indices[:, :k]
            candidate_labels = query_labels[top_k_indices]
            correct = (candidate_labels == query_labels.unsqueeze(1))

            if self.compute_per_class:
                for cls in torch.unique(query_labels):
                    cls = cls.item()
                    cls_mask = query_labels == cls
                    cls_correct = correct[cls_mask]

                    for metric in self.metrics:
                        if metrics_per_class.get(f'{metric}@{k}') is None:
                            metrics_per_class[f'{metric}@{k}'] = {}
                        metrics_per_class[f'{metric}@{k}'][cls] = self.metrics[metric](cls_correct)

            # Compute overall metrics
            for metric in self.metrics:
                metrics[f'{metric}@{k}'] = self.metrics[metric](correct)

        return metrics, metrics_per_class


def plot_metrics(
        metrics: Dict[str, float],
        metrics_per_class: Dict[str, Dict[str, float]],
        out_dir: str,
        out_filename: str,
    ):
    """Plot metrics for each class and overall metrics.

    Args:
        metrics (dict): Dictionary containing overall metrics
        metrics_per_class (dict): Dictionary containing metrics per class
    """
    os.makedirs(out_dir, exist_ok=True)

    for metric_name, class_metrics in metrics_per_class.items():
        sns.set_palette("deep")
        sns.set_style("whitegrid")
        values = list(class_metrics.values())
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=values,
            bins=10,
            stat="proportion",
            common_norm=True,
            binrange=(0, 1),
            palette=["green"]
        )
        plt.axvline(
            x=metrics[metric_name],
            color='orange',
            linestyle='--',
            label=f'Average: {metrics[metric_name]:.3f}'
        )
        plt.legend(fontsize=12)
        plt.title(f'Distribution of {metric_name} across classes', fontsize=14)
        plt.xlabel(metric_name, fontsize=12)
        plt.ylabel('Percentage of instances', fontsize=12)
        plt.ylim(0, 0.5)
        plt.xlim(0, 1.05)
        plt.savefig(
            os.path.join(
                out_dir,
                f'{metric_name}_{out_filename}.png'
            ),
            bbox_inches='tight'
        )
        plt.close()
