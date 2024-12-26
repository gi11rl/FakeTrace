from . import *
from .noise_layers import *


class Random_Noise(nn.Module):

    def __init__(self, layers, len_layers_G, len_layers_A):
        super(Random_Noise, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.noise = nn.Sequential(*layers)
        self.len_layers_G = len_layers_G
        self.len_layers_A = len_layers_A
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]
        forward_image = image.clone().detach()
        forward_cover_image = cover_image.clone().detach()
        forward_mask = mask.clone().detach()
        noised_image_G = torch.zeros_like(forward_image)
        noised_image_A = torch.zeros_like(forward_image)    
        identity_image = torch.zeros_like(forward_image)
        
        for index in range(forward_image.shape[0]):
            random_noise_layer_G = np.random.choice(self.noise[0 : self.len_layers_G], 1)[0]
            random_noise_layer_A = np.random.choice(self.noise[self.len_layers_G : self.len_layers_G + self.len_layers_A], 1)[0] 
            noised_image_G[index] = random_noise_layer_G([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0), forward_mask[index].clone().unsqueeze(0)])
            noised_image_A[index] = random_noise_layer_A([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0), forward_mask[index].clone().unsqueeze(0)])                    
            identity_image[index] = forward_image[index].clone().unsqueeze(0)

        noised_image_gap_G = noised_image_G.clamp(-1, 1) - forward_image
        noised_image_gap_A = noised_image_A.clamp(-1, 1) - forward_image
        identity_image_gap = identity_image.clamp(-1, 1) - forward_image
        
        return image + noised_image_gap_G, image + noised_image_gap_A, image + identity_image_gap