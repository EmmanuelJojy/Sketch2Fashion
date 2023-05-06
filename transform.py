from collections import OrderedDict
import torch
from PIL import Image
import torchvision.transforms as transforms
from networks import define_G

# Define the path to your pre-trained model
model_path = "model.pth"

# Define the device to use for computations
# device = torch.device('cpu')

# Load the pre-trained model
model_dict = torch.load(model_path)
new_dict = OrderedDict()
for k, v in model_dict.items():
    # load_state_dict expects keys with prefix 'module.'
    new_dict[k] = v

# make sure you pass the correct parameters to the define_G method
generator_model = define_G(input_nc=3, output_nc=3, ngf=64, netG="resnet_9blocks",
                           norm="instance", use_dropout=False, init_gain=0.02, gpu_ids=[])
generator_model.load_state_dict(new_dict)

# Set the model to evaluation mode
generator_model.eval()

# Define a data transformation pipeline for your input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def sketch2fashion(input_image_path, output_image_path):
    input_image = Image.open('static/' + input_image_path)

    # Preprocess the input image
    input_tensor = transform(input_image).unsqueeze(0)

    # Pass the input image through the generator model
    with torch.no_grad():
        output_tensor = generator_model(input_tensor)

    # Postprocess the output image
    output_image = transforms.functional.to_pil_image(
        output_tensor.squeeze().cpu())
    output_image.save('static/' + output_image_path)


# src = "testA.jpg"
# tar = "testB.jpg"
# sketch2fashion(src, tar)
