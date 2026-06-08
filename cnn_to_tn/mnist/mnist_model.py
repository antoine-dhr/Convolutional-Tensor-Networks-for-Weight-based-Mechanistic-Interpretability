from utils.model import BaseCNN

class CNNMNIST(BaseCNN):
    def __init__(self, config):
        super().__init__(
            config=config,
            in_channels=1,
            input_size=28,
            num_classes=10
        )
