import yaml
from easydict import EasyDict
from network.Dual_Mark import Network
import os
import random
import string
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import reedsolo

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_path(path="temp/"):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, ''.join(random.choices(string.ascii_letters + string.digits, k=16)) + ".png")

def string_to_message(message_str, message_length, message_range):
    byte_message = message_str.encode('utf-8')
    bit_message = ''.join(format(byte, '08b') for byte in byte_message)
    bit_message = bit_message[:message_length]
    bit_message = bit_message.ljust(message_length, '0')
    message = np.array([message_range if bit == '1' else -message_range for bit in bit_message], dtype=np.float32)
    message_tensor = torch.tensor(message).unsqueeze(0)
    return message_tensor

def encode_watermark(image, network, message, strength_factor, device):
    network.encoder_decoder.eval()
    network.discriminator.eval()

    with torch.no_grad():
        images, messages = image.to(device), message.to(device)

        encoded_images = network.encoder_decoder.module.encoder(images, messages)
        encoded_images = images + (encoded_images - images) * strength_factor

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for index in range(encoded_images.shape[0]):
            single_image = ((encoded_images[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            file = get_path()
            while os.path.exists(file):
                file = get_path()
            im.save(file)
            read = np.array(Image.open(file), dtype=np.uint8)
            os.remove(file)

            encoded_images[index] = transform(read).unsqueeze(0).to(device)

    return encoded_images

def main():
    seed_torch(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('cfg/FakeTrace_enc.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    result_folder = os.path.join("results", args.result_folder)
    model_epoch = args.model_epoch
    strength_factor = args.strength_factor
    input_image_path = args.input_image_path
    output_image_path = args.output_image_path
    watermark = args.watermark

    with open(os.path.join(result_folder, 'FakeTrace_train.yaml'), 'r') as f:
        train_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    lr = train_args.lr
    beta1 = train_args.beta1
    image_size = train_args.image_size
    message_length = train_args.message_length
    attention_encoder = train_args.attention_encoder
    attention_decoder = train_args.attention_decoder
    weight = train_args.weight
    message_range = train_args.message_range

    # 네트워크 초기화 및 모델 로드
    network = Network(
        message_length, 
        train_args.noise_layers.pool_G, 
        train_args.noise_layers.pool_A,
        device, 
        batch_size=1,  # 단일 이미지 처리
        lr=lr, 
        beta1=beta1, 
        attention_encoder=attention_encoder, 
        attention_decoder=attention_decoder, 
        weight=weight
    )
    EC_path = os.path.join(result_folder, "models", f"EC_{model_epoch}.pth")
    network.load_model_ed(EC_path)
    
    image = Image.open(input_image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)


    message_tensor = string_to_message(watermark, message_length, message_range).to(device)


    # 워터마크 인코딩
    encoded_images = encode_watermark(
        image=image_tensor,
        network=network,
        message=message_tensor,
        strength_factor=strength_factor,
        device=device
    )

    # 인코딩된 이미지 후처리 및 저장
    encoded_image = encoded_images[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
    encoded_image = ((encoded_image + 1) / 2 * 255).astype(np.uint8)
    encoded_image_pil = Image.fromarray(encoded_image)
    encoded_image_pil.save(output_image_path)
    print(f"인코딩된 이미지가 '{output_image_path}'로 저장되었습니다.")
    
    
if __name__ == '__main__':
    main()
