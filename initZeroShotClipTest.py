import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

image_names = ["Brown_Cardboard/dense_cardboard_0_crop_2925.png","Crushed_Bottles/dense_plastics_0_crop_2518.png", "Intact_Cans/dense_cans_0_crop_1522.png"]

for i in range(0,len(image_names)):
    image = preprocess(Image.open("data/Crops_Dataset/" + image_names[i])).unsqueeze(0).to(device)
    text = clip.tokenize(["cardboard", "bottle", "can"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)