import torch
from torchvision import transforms
from PIL import Image
from train import SimpleCNN

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model = SimpleCNN()
    model.load_state_dict(torch.load("./models/cifar10_cnn.pth"))
    model.eval()

    output = model(image)
    _, predicted = torch.max(output, 1)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return classes[predicted.item()]

if __name__ == '__main__':
    img_path = "test_image.png"
    print(f"Predicted: {predict(img_path)}")