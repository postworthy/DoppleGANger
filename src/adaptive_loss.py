import tensorflow as tf

class AdaptiveIdLossImprovement(tf.keras.callbacks.Callback):
    def __init__(self, improvement_threshold=0.0001, patience=3, increase_factor=1.1):
        super().__init__()
        self.improvement_threshold = improvement_threshold
        self.patience = patience
        self.increase_factor = increase_factor
        self.cumulative_improvement = 0.0
        self.wait = 0
        self.prev_id_loss = None

    def on_epoch_begin(self, epoch, logs=None):
        print(f"### AdaptiveIdLossImprovement on_epoch_begin w/ model.lambda_id: {self.model.lambda_id}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_id_loss = logs.get('id_loss')
        if current_id_loss is None:
            print("### id_loss not available in logs; skipping λ_id update.")
            return

        if self.prev_id_loss is None:
            self.prev_id_loss = current_id_loss
            return

        # Calculate improvement for this epoch
        improvement = self.prev_id_loss - current_id_loss
        self.cumulative_improvement += improvement
        self.wait += 1

        print(f"### Epoch {epoch+1}: improvement = {improvement:.7f}, cumulative improvement = {self.cumulative_improvement:.7f}")

        if self.wait >= self.patience:
            if self.cumulative_improvement < self.improvement_threshold:
                new_lambda = self.model.lambda_id * self.increase_factor
                self.model.lambda_id = new_lambda
                print(f"### No sufficient cumulative id_loss improvement in {self.patience} epochs. Increasing λ_id to {new_lambda:.7f}.")
            else:
                print("### Sufficient cumulative improvement achieved; not adjusting λ_id.")
            # Reset cumulative improvement and wait counter
            self.cumulative_improvement = 0.0
            self.wait = 0

        self.prev_id_loss = current_id_loss
