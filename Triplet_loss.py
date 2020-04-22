import tensorflow as tf
class Triplet_loss():
    def __init__(self):
        self.loss=None
    def triplet_loss(self,y_true, y_pred,  alpha=0.2):
        """
        Implementation of the triplet loss as defined by formula (3)

        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor images, of shape (None, 128)
                positive -- the encodings for the positive images, of shape (None, 128)
                negative -- the encodings for the negative images, of shape (None, 128)

        Returns:
        loss -- real number, value of the loss
        :rtype: object
        """
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        #print("pos_dist.shape", pos_dist.shape)
        # Step 2: Compute the (encoding) distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        #print("neg_dist.shape", neg_dist.shape)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        #print("basic_loss.shape", basic_loss.shape)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        self.loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
        #print("loss.shape", self.loss.shape)
        #print("--loss is ---",self.loss)
        return self.loss
    def __repr__(self):
        return f'{self.loss}'
