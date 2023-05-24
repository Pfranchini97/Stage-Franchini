import numpy as np

def emb_dist(x, y):
    #distance between embedding results, depending on the output of the network
    return np.linalg.norm(x-y)



#calculate the loss of a single pair of embedding given the correct phylogenetic distance
def hdepp_loss_couple(emb_x, emb_y, phylo_dist_xy):
    
    #emb_dist can be change accordingly to the actual input
    return np.square((np.abs(emb_dist(emb_x, emb_y)) / phylo_dist_xy) - 1)

#@
def hdepp_loss(embeddings, labels, reference):
    cost = 0
    labeled_embs = [[label, emb]for emb in embeddings for label in labels]
    
    for node1 in labeled_embs:
        for node2 in labeled_embs:
            if node1[0] == node2[0]:
                cost += 0
                continue
            cost += hdepp_loss_couple(node1[1], node2[1], reference[node1[0]][node2[0]]) 
            #depends on how the correct phylogenetic distances are stored
    return cost
