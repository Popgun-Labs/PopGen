class EMA:
    def __init__(self, decay=0.9999):
        """
        Exponential moving average /  "Polyak Averaging" for PyTorch models.
        Maintains a shadow copy of all model weights, which are updated according to a moving average
        scheme after each optimiser step.

        References:
            - https://github.com/r9y9/wavenet_vocoder/blob/f5a35bc1706a01490f59495d9e08245fab9b4de8/train.py
            - https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4

        Example usage:
            TODO

        :param decay: higher values update the shadow weights more slowly
        """
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta

    def __getitem__(self, name):
        return self.shadow[name]
