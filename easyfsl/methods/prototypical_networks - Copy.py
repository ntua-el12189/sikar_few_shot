from torch import Tensor

from .few_shot_classifier import FewShotClassifier


class PrototypicalNetworks(FewShotClassifier):

    def forward( self, query_images: Tensor,) -> Tensor:
        query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
