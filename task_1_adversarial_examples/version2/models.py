import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# Constants for ImageNet models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class EnsembleModel(nn.Module):
    """
    A robust ensemble wrapper that:
    1. Takes raw [0, 1] images of size (B, 3, 28, 28).
    2. Automatically resizes them to (224, 224) for ImageNet models.
    3. Normalizes them using ImageNet statistics.
    4. Aggregates logits from multiple architectures.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList()
        self.model_weights = []
        self.is_native = [] # Track which models need resizing
        
        # Standard ImageNet Normalization
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        # Upsampler for Group A models
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def add_imagenet_model(self, arch_name, weight=1.0):
        """Adds a standard ImageNet model (Group A: The Giants)."""
        print(f"  [+] Loading {arch_name}...", end=' ', flush=True)
        try:
            if arch_name == 'resnet50':
                m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            elif arch_name == 'densenet121':
                m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            elif arch_name == 'vgg16_bn':
                m = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
            elif arch_name == 'efficientnet_b0':
                m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            else:
                raise ValueError(f"Unknown architecture: {arch_name}")
            
            m.eval()
            m.to(self.device)
            self.models.append(m)
            self.model_weights.append(weight)
            self.is_native.append(False) # Needs upsampling
            print("Done.")
        except Exception as e:
            print(f"Failed: {e}")

    def add_native_model(self, model_instance, weight=1.0):
        """Adds a model that accepts native 28x28 or 32x32 inputs (Group B)."""
        print(f"  [+] Loading Custom Native Model...", end=' ', flush=True)
        model_instance.eval()
        model_instance.to(self.device)
        self.models.append(model_instance)
        self.model_weights.append(weight)
        self.is_native.append(True) # No upsampling needed
        print("Done.")

    def forward(self, x):
        """
        x: Tensor (B, 3, 28, 28) in range [0, 1]
        Returns: aggregated logits
        """
        total_logits = 0
        total_weight = sum(self.model_weights)
        
        # Pre-compute the upsampled version for ImageNet models to save time
        # We optimize x (28x28), but ImageNet models need 224x224
        x_upsampled = None
        
        # Pre-compute normalized inputs
        # Note: We apply normalization AFTER resizing usually, or before? 
        # Standard PyTorch pipeline: Tensor [0,1] -> Resize -> Normalize -> Model
        
        for i, model in enumerate(self.models):
            weight = self.model_weights[i]
            
            if self.is_native[i]:
                # Native models (e.g. CIFAR-10 structure) usually expect their own normalization
                # For now assuming they take standardized inputs or we handle it here.
                # Since we don't have them yet, we treat x as the input.
                inp = self.normalize(x) # Assuming native models also trained on standard stats or similar
                logits = model(inp)
            else:
                # ImageNet Models
                if x_upsampled is None:
                    x_upsampled = self.upsample(x)
                    x_upsampled = self.normalize(x_upsampled)
                
                logits = model(x_upsampled)
            
            total_logits += logits * weight
            
        return total_logits / total_weight

