import torch
from torchvision import transforms
from model.dpfnutrition import DPFNutritionModel
from PIL import Image

image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.)
])

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DPFNutritionModel(
        device=device,
        state_dict_file="depth_model_weights_finetuned.pth",
        no_depth=True
    )

    model.load_state_dict(torch.load("dpfnutrition_1d_depth_main_model.pth", weights_only=True, map_location=device))
    model.eval() # set model to evaluation/inferece mode

    image_input = Image.open("rgb.png").convert("RGB")
    image_input = image_transform(image_input).unsqueeze(0).to(device)

    output = model(image_input)
    calories, mass, fat, carbs, protein = output[0]
    caloreis = calories.item()
    mass = mass.item()
    fat = fat.item()
    carbs = carbs.item()
    protein = protein.item()