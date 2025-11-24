# test_registry.py
from ddriver.models import registry
import torch.nn as nn

# Test 1: Custom model registration
@registry.register_model("dummy_linear")
def build_dummy(num_classes=10, in_features=4):
    return nn.Linear(in_features, num_classes)

model = registry.build_model("dummy_linear", num_classes=5, in_features=8)
print("Model:", model)
print("Available models:", registry.available_models())