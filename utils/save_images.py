'''
Function to save images

By jzyustc, 2020/12/21

'''

import os
import numpy as np
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def save_images(saved_all, epoch, folder, resize_to=None):
	original_images, watermarked_images, noised_images_G, noised_images_A = saved_all

	images = original_images[:original_images.shape[0], :, :, :].cpu()
	watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
	noised_images_G = noised_images_G[:noised_images_G.shape[0], :, :, :].cpu()
	noised_images_A = noised_images_A[:noised_images_A.shape[0], :, :, :].cpu()

	images = (images + 1) / 2
	watermarked_images = (watermarked_images + 1) / 2
	noised_images_G = (noised_images_G + 1) / 2
	noised_images_A = (noised_images_A + 1) / 2

	if resize_to is not None:
		images = F.interpolate(images, size=resize_to)
		watermarked_images = F.interpolate(watermarked_images, size=resize_to)
	stacked_images = torch.cat(
		[images.unsqueeze(0), watermarked_images.unsqueeze(0), noised_images_G.unsqueeze(0), noised_images_A.unsqueeze(0)], dim=0)
 
	shape = stacked_images.shape
	stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
	stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
	filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))

	saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
	saved_image.save(filename)


def get_random_images(images, encoded_images, noised_images_G, noised_images_A):
	selected_id = np.random.randint(1, images.shape[0]) if images.shape[0] > 1 else 1
	image = images.cpu()[selected_id - 1:selected_id, :, :, :]
	encoded_image = encoded_images.cpu()[selected_id - 1:selected_id, :, :, :]
	noised_image_G = noised_images_G.cpu()[selected_id - 1:selected_id, :, :, :]
	noised_image_A = noised_images_A.cpu()[selected_id - 1:selected_id, :, :, :]
	return [image, encoded_image, noised_image_G, noised_image_A]


def concatenate_images(saved_all, images, encoded_images, noised_images_G, noised_images_A=None):    
	saved = get_random_images(images, encoded_images, noised_images_G, noised_images_A)
	if saved_all[3].shape[2] != saved[3].shape[2]:
		return saved_all
	saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
	saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
	saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
	saved_all[3] = torch.cat((saved_all[3], saved[3]), 0)		# 추가
	return saved_all

def save_images_per_noise_layer(images, encoded_images, noised_images, folder, noise_layer_name, epoch, resize_to=None):
    """
    각 noise layer에 대해 배치의 이미지를 세로 방향으로 concatenate한 후 저장.
    """
    
    images = images[:images.shape[0], :, :, :].cpu()
    encoded_images = encoded_images[:encoded_images.shape[0], :, :, :].cpu()
    noised_images = noised_images[:noised_images.shape[0], :, :, :].cpu()
    
    images = (images + 1) / 2
    encoded_images = (encoded_images + 1) / 2
    noised_images = (noised_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        encoded_images = F.interpolate(encoded_images, size=resize_to)
        noised_images = F.interpolate(noised_images, size=resize_to)
        
    batch_images = []
    for i in range(images.size(0)):
        row = torch.cat([images[i], encoded_images[i], noised_images[i]], dim=1)
        batch_images.append(row)

    final_image = torch.cat(batch_images, dim=2)

    layer_folder = os.path.join(folder, noise_layer_name)
    os.makedirs(layer_folder, exist_ok=True)

    save_path = os.path.join(layer_folder, f"epoch_{epoch}.png")
    save_image(final_image, save_path)

def _normalize(input_tensor):
	output = input_tensor.clone()
	for i in range(output.shape[0]):
		min_val, max_val = torch.min(output[i]), torch.max(output[i])
		output[i] = (output[i] - min_val) / (max_val - min_val)

	return output

