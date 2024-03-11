import os
import secrets


class Arguments:
    def __init__(
        self,
        subj: int,
        data_dir: str,
        val_split: int,
        model: str = "alexnet",
        layers: list[str] = ["features.2"],
        roi: str | None = None,
        run_id: str = secrets.token_hex(4),
        device: str = "cpu",
    ):
        self.subj: str = format(subj, "02")
        self.data_dir = os.path.join(data_dir, "subj" + self.subj)

        self.val_split = val_split
        assert 0 < self.val_split < 100, "val_split must be strictly between 0 and 100"

        self.model = model
        self.layers = layers
        self.roi = roi

        self.run_id = run_id

        self.device = device
