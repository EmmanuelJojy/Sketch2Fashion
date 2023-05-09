from collections import OrderedDict
import torch
from PIL import Image
import torchvision.transforms as transforms

from lib.networks import define_G
from lib.util import tensor2im, save_image

models = {
    'resnet': {'path': 'models/cyclegan-resnet.pth', 'input_nc': 3, 'output_nc': 3, 'netG': "resnet_9blocks", 'norm': "instance", },
    'unet': {'path': 'models/cyclegan-unet.pth', 'input_nc': 3, 'output_nc': 3, 'netG': "unet_256", 'norm': "instance", },
    'pix2pix': {'path': 'models/pix2pix.pth', 'input_nc': 3, 'output_nc': 3, 'netG': "unet_256", 'norm': "batch", },
    'color': {'path': 'models/pix2pix-color.pth', 'input_nc': 1, 'output_nc': 2, 'netG': "unet_256", 'norm': "instance", },
}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def create_generator(model):
    model_dict = torch.load(model['path'])
    new_dict = OrderedDict()
    for k, v in model_dict.items():
        new_dict[k] = v
    generator_model = define_G(input_nc=model['input_nc'], output_nc=model['output_nc'], netG=model['netG'],
                               norm=model['norm'], ngf=64, use_dropout=False, init_gain=0.02, gpu_ids=[])
    generator_model.load_state_dict(new_dict)

    # Set the model to evaluation mode
    generator_model.eval()

    return generator_model


def sketch2fashion(generator_model, input_image_path='test.jpg', output_image_path='out.jpg'):
    input_image = Image.open(input_image_path).convert('RGB')
    image_size = input_image.size

    # Preprocess the input image
    input_tensor = transform(input_image).unsqueeze(0)

    # Pass the input image through the generator model
    with torch.no_grad():
        output_tensor = generator_model(input_tensor)

    # Postprocess the output image
    output_image = tensor2im(output_tensor)
    save_image(output_image, output_image_path, image_size)


# set up generators
resnet = create_generator(models['resnet'])
print("CycleGAN (Gen: Resnet): Ready", end='\n\n')

unet = create_generator(models['unet'])
print("CycleGAN (Gen: U -NET): Ready", end='\n\n')

pix2pix = create_generator(models['pix2pix'])
print("Pix 2 Pix (Gen: U NET): Ready", end='\n\n')

# color = create_generator(models['color'])
# print("Pix 2 Pix (Var: Color): Ready", end='\n\n')


sketch2fashion(resnet, input_image_path='test.jpg', output_image_path='resnet.jpg')
sketch2fashion(resnet, input_image_path='test.jpg', output_image_path='unet.jpg')
sketch2fashion(resnet, input_image_path='test.jpg', output_image_path='pix2pix.jpg')
