class ImportanceScoreCalculator:
    def __init__(self, weights, bias, last_layer_imp_scores):
        self.weights = weights
        self.bias = bias
        self.last_layer_imp_scores = last_layer_imp_scores
        self.nodes = weights.shape[0]
        self.nconnections = weights.shape[1]

    def print_info(self):
        print("number of nodes :", self.nodes)

    # MLP model
    def calculate_imp_score(self):
        imp_score_list = list()

        for i in range(self.nodes):
            imp_score = 0

            for j in range(self.nconnections):
                ####
                # Score (Si) =  E |Wij| Sj
                ####
                imp_score += (abs(self.weights[i, j]) * self.last_layer_imp_scores[j])

            imp_score_list.append(imp_score)

        return imp_score_list


# CNN model
class CnnFilterScoreCalculator(ImportanceScoreCalculator):

    def calculate_imp_score(self):
        constant, self.nodes, self.nconnections = self.weights.shape
        print("Nodes:", self.nodes, ", Filters: ", self.nconnections)
        return abs(self.weights) * self.last_layer_imp_scores
