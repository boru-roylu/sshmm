from utils import (
    graph_topo,
)
import pickle
import pdb

class GraphTopo:
    def __init__(self):
        self.load_model()
        self.graph_transmat()


    def load_model(self):
        with open('./exp/models_26/6.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def graph_transmat(self):
        print(self.model.ordered_transmat)
        print(self.model.state_transmat_info)
        graph = graph_topo(self.model.ordered_transmat, self.model.state_transmat_info)


if __name__ == '__main__':
    GraphTopo()
