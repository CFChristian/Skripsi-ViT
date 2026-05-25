import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# LOAD MODEL
def load_model_10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    num_classes = 10
    in_features = model.heads.head.in_features

    model.heads.head = nn.Sequential(
        nn.Dropout(0.18902495801181993),
        nn.Linear(in_features, num_classes)
    )

    checkpoint = torch.load("model_vit_10_classes_aug_fix.pth", map_location=device)

    model.load_state_dict(checkpoint['model_state_dict_aug_fix'])
    model.to(device)
    model.eval()

    class_names = checkpoint['class_names_aug_fix']

    return model, class_names, device

def load_model_5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    num_classes = 5
    in_features = model.heads.head.in_features

    model.heads.head = nn.Sequential(
        nn.Dropout(0.1806109739953483),
        nn.Linear(in_features, num_classes)
    )

    checkpoint = torch.load("model_vit_5_classes_aug_fix.pth", map_location=device)

    model.load_state_dict(checkpoint['model_state_dict_aug_fix'])
    model.to(device)
    model.eval()

    class_names = checkpoint['class_names_aug_fix']

    return model, class_names, device

# PREDICT
def predict_image(model, image, device):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    return probs.cpu().numpy()[0]
