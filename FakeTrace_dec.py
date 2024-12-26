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

def string_to_message(message_str, message_length, message_range):
    #byte_message = message_str.encode('utf-8')
    #bit_message = ''.join(format(byte, '08b') for byte in byte_message)
    bit_message = ''.join(format(byte, '08b') for byte in message_str)
    bit_message = bit_message[:message_length]
    bit_message = bit_message.ljust(message_length, '0')
    message = np.array([message_range if bit == '1' else -message_range for bit in bit_message], dtype=np.float32)
    message_tensor = torch.tensor(message).unsqueeze(0)
    return message_tensor

def decoded_message_error_rate(message, decoded_message):
    length = message.shape[1]  # message는 [batch_size, length] 형태임
    # 두 텐서를 동일한 디바이스로 이동
    decoded_message = decoded_message.to(message.device)
    
    message = message.gt(0)
    decoded_message = decoded_message.gt(0)
    error_rate = float((message != decoded_message).sum()) / length
    return error_rate

def decode_message(decoded_tensor, message_length):
    binary_message = ""
    decoded_tensor = decoded_tensor.squeeze(0).cpu().numpy()
    for i in range(message_length):
        binary_message += "1" if decoded_tensor[i] > 0 else "0"
    ascii_message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        if len(byte) == 8:  # 8비트가 채워졌을 때만 변환
            ascii_message += chr(int(byte, 2))
    return ascii_message


def decode_watermark(encoded_image, network, device):
    network.encoder_decoder.eval()
    network.discriminator.eval()

    with torch.no_grad():
        images = encoded_image.to(device)

        decoded_messages_G = network.encoder_decoder.module.decoder_G(images)
        decoded_messages_A = network.encoder_decoder.module.decoder_A(images)

    return decoded_messages_G, decoded_messages_A

def main():
    seed_torch(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('cfg/FakeTrace_dec.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    result_folder = os.path.join("results", args.result_folder)
    model_epoch = args.model_epoch
    input_image_path = args.input_image_path
    extracted_watermark_path = args.extracted_watermark_path

    with open(os.path.join(result_folder, 'FakeTrace_train.yaml'), 'r') as f:
        train_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    lr = train_args.lr
    beta1 = train_args.beta1
    image_size = train_args.image_size
    message_length = train_args.message_length
    attention_encoder = train_args.attention_encoder
    attention_decoder = train_args.attention_decoder
    weight = train_args.weight

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

    # 이미지 로드 및 전처리
    image = Image.open(input_image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 필요 시 이미지 크기 조정
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # 배치 차원 추가

    # 워터마크 디코딩
    decoded_messages_G, decoded_messages_A = decode_watermark(
        encoded_image=image_tensor,
        network=network,
        device=device
    )
    
    #message_tensor = string_to_message("hellowatermark", message_length, 1)
    message_tensor = string_to_message(b'hi\x16MMS\xfa\x14\x0f\xcc\x1b%\xd0J\xc1\xd947\xc7(VLPxT\xfb\\\xd5n\xa6\\U\xeaoH,<\xcd\xc8\t\x12)\xfe\xb4{;9\xcc\xd5\x190\x1cG\x83\xa5\xb4*\xbe\x1e\xd7\xbf\xf8\x98[\xcc[\xack.\xd3\x1d\xbb$\xf8~.U\x1b\x80\xa7\x8fe\xb49\x9a0\x94\xa9\xc4\x98\xb3\x0b<2\xfb\xc8\xe9:.\xe2\x99ZM\xab\xfe\x92,|s;Am\xbe\x0e',
                                       message_length, 1)
    error_rate_G = decoded_message_error_rate(message_tensor ,decoded_messages_G)
    error_rate_A = decoded_message_error_rate(message_tensor ,decoded_messages_A)

    # 메시지 변환
    decoded_message_G = decode_message(decoded_messages_G, message_length)
    decoded_message_A = decode_message(decoded_messages_A, message_length)

    
    # 디코딩된 메시지 저장
    with open(extracted_watermark_path, 'w') as f:
        f.write("Decoded Message G: " + decoded_message_G + "\n")
        f.write("Error_Rate_G: " + str(float(error_rate_G)) + "\n\n")
        f.write("Decoded Message A: " + decoded_message_A + "\n")
        f.write("Error_Rate_A: " + str(float(error_rate_A)) + "\n\n")

    #print(f"디코딩된 메시지가 '{extracted_watermark_path}'에 저장되었습니다.")
    print("Decoded Message G:", decoded_message_G)
    print("Decoded Message A:", decoded_message_A)
    

if __name__ == '__main__':
    main()
