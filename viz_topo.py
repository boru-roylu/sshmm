from utils import (
    graph_topo,
)
import pickle
import pdb

class GraphTopo:
    def __init__(self, n_states):
        self.top_e = 5
        self.n_states = n_states
        self.load_model()
        self.graph_transmat()


    def load_model(self):
        with open(f'./exp/models_30_topk_3/{self.n_states}.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def graph_transmat(self):
        print(self.model.ordered_transmat)
        print(self.model.state_transmat_info)
        vocab = {y:x for x,y in self.model.vocab.items()}
        dot = graph_topo(self.model.transmat_, self.model.emissionprob_,
                           self.model.state_transmat_info, vocab, self.top_e)
        dot.render(f'./topology/sshmm/sshmm_{self.n_states}states', format='pdf', view=False)


if __name__ == '__main__':

    for num_s in range(3, 16):
        GraphTopo(num_s)
