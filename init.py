# Pseudo-code for on-device training
class TinyGazeTrainer:
    def __init__(self):
        self.model = load_pretrained_model()
        self.freeze_backbone()
        
    def freeze_backbone(self):
        # Only train final layers
        for layer in self.model.layers[:-3]:
            layer.trainable = False
    
    def train_step(self, eye_image, target_gaze):
        # Forward pass with checkpointing
        features = self.model.backbone(eye_image)
        prediction = self.model.head(features)
        
        # Compute loss
        loss = mse_loss(prediction, target_gaze)
        
        # Backward pass (only for head layers)
        gradients = compute_gradients(loss, trainable_vars)
        
        # Update with small learning rate
        apply_gradients(gradients, lr=0.0001)
        
        return loss
    
    def online_learning(self, calibration_data):
        for epoch in range(5):  # Few epochs
            for eye_img, gaze in calibration_data:
                loss = self.train_step(eye_img, gaze)
