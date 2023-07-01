import os, shutil
from collections import OrderedDict
import torch
from PIL import Image
import torchvision.transforms as transforms

from lib.networks import define_G, define_D
from lib.util import tensor2im, save_image

models = {
    'unet-new': {'path': 'models/cyclegan-resnet.pth', 'netG': "resnet_9blocks", 'norm': "instance", },
    'unet': {'path': 'models/cyclegan-unet.pth', 'netG': "unet_256", 'norm': "instance", },
    'unet_new': {'path': 'models/cyclegan-unet-new.pth', 'netG': "unet_256", 'norm': "instance", },
    'pix2pix': {'path': 'models/pix2pix.pth', 'netG': "unet_256", 'norm': "batch", },
    'dis': {'path': 'models/latest_net_D_A.pth'}
}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_without_grayscale = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def create_generator(model, ngf=64):
    model_dict = torch.load(model['path'])
    new_dict = OrderedDict()
    for k, v in model_dict.items():
        new_dict[k] = v
    generator_model = define_G(input_nc=3, output_nc=3, netG=model['netG'],
                               norm=model['norm'], ngf=ngf, use_dropout=False, init_gain=0.02, gpu_ids=[])
    generator_model.load_state_dict(new_dict)

    # Set the model to evaluation mode
    generator_model.eval()

    return generator_model

def create_discriminator(model, ngf=64):
    model_dict = torch.load(model['path'])
    new_dict = OrderedDict()
    for k, v in model_dict.items():
        new_dict[k] = v
    generator_model = define_D(input_nc=3, ndf=64)
    generator_model.load_state_dict(new_dict)

    # Set the model to evaluation mode
    generator_model.eval()

    return generator_model

def sketch2fashion(generator_model, input_image_path='test.jpg', output_image_path='out.jpg', grayscale=True):
    input_image = Image.open(input_image_path).convert('RGB')
    image_size = input_image.size

    # Preprocess the input image
    if grayscale:
        input_tensor = transform(input_image).unsqueeze(0)
    else:
        input_tensor = transform_without_grayscale(input_image).unsqueeze(0)

    # Pass the input image through the generator model
    with torch.no_grad():
        output_tensor = generator_model(input_tensor)

    # Postprocess the output image
    output_image = tensor2im(output_tensor)
    save_image(output_image, output_image_path, image_size)


# set up generators
resnet = create_generator(models['unet-new'])
print("CycleGAN (Gen: UNET): Ready", end='\n\n')

# unet = create_generator(models['unet'])
# print(unet)
# print("CycleGAN (Gen: U -NET): Ready", end='\n\n')

# d_a = create_discriminator(models['dis'])
# print(d_a)



# unet_cycle = create_generator(models['unet_new'], ngf=128)
# print("CycleGAN (Gen: UNETx2): Ready", end='\n\n')


# RUNNER

# sketch2fashion(resnet, input_image_path='test.jpg', output_image_path='out_resnet.jpg')
# sketch2fashion(unet, input_image_path='test.jpg', output_image_path='out_unet.jpg')
# sketch2fashion(unet_cycle, input_image_path='test.jpg', output_image_path='out_unet_cycle.jpg')

import time

for file_name in os.listdir('images'):
    if file_name.endswith(".png"):
        file_path = 'images/' + file_name
        shutil.copyfile(file_path, 'test.jpg')
        sketch2fashion(resnet, input_image_path=file_path, output_image_path='out_final.jpg')
        # sketch2fashion(unet, input_image_path=file_path, output_image_path='out_unet.jpg')
        # sketch2fashion(pix2pix, input_image_path=file_path, output_image_path='out_pix2pix.jpg')
        time.sleep(2)
