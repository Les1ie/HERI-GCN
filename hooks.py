from dgl import DGLHeteroGraph
from torch import Tensor


class PopularityPredictorHooks:
    """
    Used to implement other `forward` computation.
    Workflow of popularity predictor forward:
     -> process_graph
     -> on_conv_start
     -> [
         -> on_conv_step_start
         -> conv
         -> on_conv_step_end
         ] * 2
     -> on_conv_end
     -> prediction
    """

    def on_conv_start(self, g: DGLHeteroGraph, feats_dim: int) -> DGLHeteroGraph:
        """
        Hook before predefined graph convolution.
        :param g: heterogeneous graph in workflow.
        :param feats_dim: feature dimension.
        :return: DGLHeteroGraph
        """
        return g

    def on_conv_end(self, g: DGLHeteroGraph, feats_dim: int) -> DGLHeteroGraph:
        """
        Hook after predefined graph convolution.
        :param g: heterogeneous graph in workflow.
        :param feats_dim: feature dimension.
        :return: DGLHeteroGraph
        """
        return g

    def on_conv_step_start(self, g: DGLHeteroGraph, feats_dim: int) -> DGLHeteroGraph:
        """
        Hook before steps in predefined graph convolution.
        :param g: heterogeneous graph in workflow.
        :param feats_dim: feature dimension.
        :return: DGLHeteroGraph
        """
        return g

    def on_conv_step_end(self, g: DGLHeteroGraph, feats_dim: int) -> DGLHeteroGraph:
        """
        Hook after steps in predefined graph convolution.
        :param g: heterogeneous graph in workflow.
        :param feats_dim: feature dimension.
        :return: DGLHeteroGraph
        """
        return g

    def on_readout_start(self, g: DGLHeteroGraph, feats_dim: int) -> DGLHeteroGraph:
        """
        Hook before graph readout for prediction.
        :param g: heterogeneous graph in workflow.
        :param feats_dim: feature dimension.
        :return: DGLHeteroGraph
        """
        return g

    def on_readout_end(self, popularity: Tensor) -> Tensor:
        """
        Hook after graph readout for prediction.
        :param popularity: result of graph readout.
        :return: Tensor
        """
        return popularity
