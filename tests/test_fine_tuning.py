from fine_tune_clean_model import configure_block5_for_fine_tuning


class FakeLayer:
    def __init__(self, name: str, has_weights: bool = True):
        self.name = name
        self.weights = [object()] if has_weights else []
        self.trainable = False


class FakeBaseModel:
    def __init__(self):
        self.trainable = False
        self.layers = [
            FakeLayer("block4_conv3"),
            FakeLayer("block4_pool", has_weights=False),
            FakeLayer("block5_conv1"),
            FakeLayer("block5_conv2"),
            FakeLayer("block5_conv3"),
            FakeLayer("block5_pool", has_weights=False),
        ]


class FakeModel:
    def __init__(self):
        self.base_model = FakeBaseModel()

    def get_layer(self, name: str):
        assert name == "vgg16"
        return self.base_model


def test_only_vgg16_block5_weights_are_unfrozen():
    model = FakeModel()

    trainable = configure_block5_for_fine_tuning(model)

    assert trainable == ["block5_conv1", "block5_conv2", "block5_conv3"]
    assert model.base_model.trainable
    assert not model.base_model.layers[0].trainable
    assert all(layer.trainable for layer in model.base_model.layers[2:])
