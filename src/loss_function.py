import numpy as np
import tensorflow as tf

def emb_dist(x, y):
    #distance between embedding results, depending on the output of the network
    return tf.math.acosh(1 + tf.math.abs(2*tf.math.divide(tf.math.square(tf.norm(x-y)),
                                                          tf.math.multiply(1 - tf.math.square(tf.norm(x)) ,
                                                                           1 - tf.math.square(tf.norm(y)) ) ) ) ) 



#calculate the loss of a single pair of embedding given the correct phylogenetic distance
def hdepp_loss_couple(emb_x, emb_y, phylo_dist_xy):
    
    #emb_dist can be change accordingly to the actual input
    return tf.math.square(tf.math.divide(tf.math.abs(emb_dist(emb_x, emb_y)) , tf.cast(phylo_dist_xy, tf.float32)) - 1)
    

#@tf.function
def hdepp_loss(embeddings, labels, reference):
    cost = 0
    #labeled_embs = [[label, emb] for emb in embeddings for label in labels]
    labeled_embs = [recon for recon in zip(labels, embeddings)]
    
    n_nodes = len(labeled_embs)
    for i in range(n_nodes):
        node1 = labeled_embs[i]
        
        for j in range(n_nodes):
            node2 = labeled_embs[j]
            node1_pos = tf.math.argmax(node1[0])
            node2_pos = tf.math.argmax(node2[0])
            
            if node1_pos == node2_pos:
                cost += 0
                continue
                
            cost += hdepp_loss_couple(node1[1], node2[1], reference[node1_pos][node2_pos]) 
            #depends on how the correct phylogenetic distances are stored
            
    return cost
