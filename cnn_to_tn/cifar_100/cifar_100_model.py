from utils.model import BaseCNN

class CNNCIFAR100(BaseCNN):
    def __init__(self, config):
        super().__init__(
            config=config,
            in_channels=3,
            input_size=32,
            num_classes=100,
        )
