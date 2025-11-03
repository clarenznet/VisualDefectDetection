# Alternative: No backpropagation needed
class ForwardForwardGaze:
    def train_layer(self, layer, positive_data, negative_data):
        # Positive pass
        pos_activity = layer(positive_data)
        pos_goodness = sum(pos_activity ** 2)
        
        # Negative pass (augmented incorrect gaze)
        neg_activity = layer(negative_data)
        neg_goodness = sum(neg_activity ** 2)
        
        # Update weights to maximize positive, minimize negative
        layer.weights += lr * (pos_goodness - neg_goodness)
