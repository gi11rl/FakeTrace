from . import *
from .Encoder_U import DW_Encoder
from .Decoder_U import DW_Decoder
# from .Noise import Noise
from .Random_Noise import Random_Noise


class DW_EncoderDecoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, message_length, noise_layers_G, noise_layers_A, attention_encoder, attention_decoder):

		super(DW_EncoderDecoder, self).__init__()
		self.encoder = DW_Encoder(message_length, attention = attention_encoder)
		self.noise = Random_Noise(noise_layers_G + noise_layers_A, len(noise_layers_G), len(noise_layers_A))

		self.decoder_G = DW_Decoder(message_length, attention = attention_decoder)
		self.decoder_A = DW_Decoder(message_length, attention = attention_decoder)


	def forward(self, image, message, mask):
		encoded_image = self.encoder(image, message)
		noised_image_G, noised_image_A, identity_image = self.noise([encoded_image, image, mask])
		
		decoded_message_G = self.decoder_G(noised_image_G)
		decoded_message_A = self.decoder_A(noised_image_A)
  
		decoded_message_G_I = self.decoder_G(identity_image)
		decoded_message_A_I = self.decoder_A(identity_image)
		return encoded_image, noised_image_G, noised_image_A, decoded_message_G, decoded_message_A, decoded_message_G_I, decoded_message_A_I

