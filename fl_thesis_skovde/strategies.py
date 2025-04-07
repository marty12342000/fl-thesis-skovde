from .task import Net, set_weights
import torch
import wandb
from datetime import datetime
from logging import INFO, WARNING
from typing import Callable, Optional, Union, Dict, Tuple, List

from flwr.common import (
    EvaluateIns, # Ajout de EvaluateIns pour la complétude
    EvaluateRes,
    FitRes, 
    MetricsAggregationFn, 
    NDArrays, 
    Parameters, 
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters # Ajout pour la complétude si nécessaire
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager # Ajout pour la complétude si nécessaire
from flwr.server.strategy import FedAvg




#class CustomStrategy(FedAvg):
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
#        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#        wandb.init(project="fl-thesis-skovde", name=f"custom-strategy-{name}")
#
#
#    def aggregate_fit(self, 
#                      server_round: int, 
#                      results: list[tuple[ClientProxy, FitRes]], 
#                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
#                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
#        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
#
#        # convert parameters to ndarrays
#        ndarrays = parameters_to_ndarrays(parameters_aggregated)
#
#        # instantiate model
#        model = Net()
#        set_weights(model, ndarrays)
#
#        # save model
#        torch.save(model.state_dict(), f"global_model_round_{server_round}.pth")
#
#
#        return parameters_aggregated, metrics_aggregated
#   
#
#    def evaluate(self, 
#                 server_round: int, 
#                 parameters: Parameters
#                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
#        
#        loss, metrics = super().evaluate(server_round, parameters)
#
#        my_results = {"loss": loss, **metrics}
#
#        wandb.log(my_results, step=server_round)
#
#
#        return loss, metrics
    


class EarlyStoppingAMBS(FedAvg):
    """
    Stratégie FedAvg personnalisée avec arrêt anticipé (Early Stopping).

    Cette stratégie arrête l'entraînement fédéré si une métrique surveillée
    ne s'améliore pas pendant un nombre défini de tours ('patience').

    Args:
        patience (int): Nombre de tours sans amélioration avant de demander l'arrêt.
        min_delta (float): Changement minimal dans la métrique surveillée pour
                           qualifier comme une amélioration.
        metric_to_monitor (str): Nom de la métrique à surveiller dans le dict
                                 de métriques retourné par aggregate_evaluate.
                                 Utilisez 'loss' pour surveiller la perte agrégée.
        mode (str): 'min' ou 'max'. En mode 'min', l'entraînement s'arrête quand
                    la métrique cesse de diminuer. En mode 'max', il s'arrête
                    quand la métrique cesse d'augmenter.
        *args: Arguments à passer au constructeur FedAvg parent.
        **kwargs: Arguments nommés à passer au constructeur FedAvg parent.
    """
    def __init__(
        self,
        *, # Force les arguments suivants à être des arguments nommés
        patience: int = 1,           # Nombre de tours d'attente par défaut
        min_delta: float = 0.001,    # Différence minimale pour considérer une amélioration
        metric_to_monitor: str = "loss", # Surveiller la perte par défaut
        mode: str = "min",           # La perte doit diminuer
        **kwargs # Passe les autres arguments à FedAvg
    ):
        super().__init__(**kwargs) # Passe les arguments pertinents à FedAvg

        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project="fl-thesis-skovde", name=f"custom-strategy-{name}")

        if mode not in ["min", "max"]:
            raise ValueError("Le mode doit être 'min' ou 'max'")
        if patience < 1:
            raise ValueError("La patience doit être d'au moins 1")

        self.patience = patience
        self.min_delta = min_delta
        self.metric_to_monitor = metric_to_monitor
        self.mode = mode

        # Variables d'état pour l'arrêt anticipé
        self.rounds_without_improvement = 0
        if self.mode == "min":
            self.best_metric = float('inf') # Initialise au pire pour la minimisation
        else: # mode == "max"
            self.best_metric = float('-inf') # Initialise au pire pour la maximisation

        self._stop_requested = False # Indicateur interne pour l'arrêt

        log(INFO, f"Stratégie EarlyStoppingAMBS initialisée : "
                  f"patience={self.patience}, min_delta={self.min_delta}, "
                  f"metric='{self.metric_to_monitor}', mode='{self.mode}'")

    def should_stop(self) -> bool:
        """
        Indique si l'entraînement doit s'arrêter selon les critères d'arrêt anticipé.

        Returns:
            bool: True si l'arrêt est demandé, False sinon.
        """
        return self._stop_requested

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Agrège les résultats d'évaluation et vérifie les conditions d'arrêt anticipé."""

        # 1. Agrégation standard (appel à la méthode parente FedAvg)
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # 2. Vérification de l'arrêt anticipé (si l'évaluation a réussi)
        if aggregated_loss is not None:
            current_metric = None
            if self.metric_to_monitor == "loss":
                current_metric = aggregated_loss
            elif self.metric_to_monitor in aggregated_metrics:
                current_metric = aggregated_metrics[self.metric_to_monitor]
            else:
                log(WARNING, f"EarlyStopping: Métrique '{self.metric_to_monitor}' "
                             f"non trouvée dans les métriques agrégées ({list(aggregated_metrics.keys())}). "
                             f"Impossible de vérifier l'arrêt anticipé pour ce tour.")

            if current_metric is not None:
                # Vérifie l'amélioration basée sur le mode ('min' ou 'max')
                is_improvement = False
                if self.mode == "min":
                    # Amélioration si la métrique actuelle est inférieure à la meilleure - delta
                    if current_metric < self.best_metric - self.min_delta:
                        is_improvement = True
                else: # mode == "max"
                    # Amélioration si la métrique actuelle est supérieure à la meilleure + delta
                    if current_metric > self.best_metric + self.min_delta:
                        is_improvement = True

                # Mise à jour de l'état
                if is_improvement:
                    self.best_metric = current_metric
                    self.rounds_without_improvement = 0
                    log(INFO, f"EarlyStopping: Amélioration détectée au tour {server_round}. "
                              f"Nouvelle meilleure métrique ({self.metric_to_monitor}): {self.best_metric:.4f}")
                else:
                    self.rounds_without_improvement += 1
                    log(INFO, f"EarlyStopping: Pas d'amélioration significative au tour {server_round}. "
                              f"Compteur de patience: {self.rounds_without_improvement}/{self.patience}")

                # Vérification de la condition d'arrêt
                if self.rounds_without_improvement >= self.patience:
                    self._stop_requested = True
                    log(INFO, f"EarlyStopping: Condition d'arrêt remplie au tour {server_round}. "
                              f"Patience ({self.patience}) atteinte.")
            else:
                 # Si la métrique n'a pas pu être récupérée, on ne peut pas vérifier
                 log(WARNING, f"EarlyStopping: Impossible de récupérer la métrique '{self.metric_to_monitor}' "
                              f"pour le tour {server_round}. La vérification est ignorée.")


        # 3. Retourner les résultats agrégés
        return aggregated_loss, aggregated_metrics

    # Vous pouvez garder ou ajouter d'autres méthodes surchargées ici si nécessaire
    # Par exemple, si vous voulez aussi logger avec wandb ou sauvegarder le modèle
    # comme dans CustomStrategy, vous pouvez ajouter ces méthodes ici aussi.
    # N'oubliez pas d'appeler super() si vous surchargez __init__.

    # Exemple : Ajout du logging wandb comme dans CustomStrategy
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Évalue les paramètres (potentiellement côté serveur) et logue avec wandb."""
        # Note: Cette méthode est pour l'évaluation côté serveur (si evaluate_fn est fourni).
        # L'évaluation fédérée (clients) passe par aggregate_evaluate.
        # Adaptez selon que vous utilisez evaluate_fn ou l'évaluation fédérée.

        eval_results = super().evaluate(server_round, parameters)

        if eval_results:
            loss, metrics = eval_results
            my_results = {"server_eval_loss": loss, **{f"server_{k}": v for k,v in metrics.items()}}
            # Utilisez un préfixe pour éviter les conflits si aggregate_evaluate logue aussi
            try:
                if wandb.run:
                     wandb.log(my_results, step=server_round)
            except Exception as e:
                 log(WARNING, f"Wandb logging in evaluate failed: {e}")
            return loss, metrics
        return None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        """Log into wandb"""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if metrics_aggregated:
             try:
                 if wandb.run:
                      wandb.log({f"train_{k}": v for k,v in metrics_aggregated.items()}, step=server_round)
             except Exception as e:
                 log(WARNING, f"Wandb logging in aggregate_fit failed: {e}")


        return parameters_aggregated, metrics_aggregated


    

    





