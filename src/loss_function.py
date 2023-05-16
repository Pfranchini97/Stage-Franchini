import numpy as np

def emb_dist(x, y):
    #distance between embedding results, depending on the output of the network
    return x-y



#calculate the loss of a single pair of embedding given the correct phylogenetic distance
def hdepp_loss_couple(emb_x, emb_y, phylo_dist_xy):
    
    #emb_dist can be change accordingly to the actual input
    return np.square((np.abs(emb_dist(emb_x, emb_y)) / phylo_dist_xy) - 1)


def hdepp_loss(embeddings, phylo_distances):
    cost = 0
    for e1 in embeddings:
        for e2 in embeddings:
            cost += hdepp_loss_couple(e1, e2, phylo_distances[e1][e2]) #depends on how the correct phylogenetic distances are stored
    return cost
