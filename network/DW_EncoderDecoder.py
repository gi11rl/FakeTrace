from . import *
from .Encoder_U import DW_Encoder
from .Decoder_U import DW_Decoder
# from .Noise import Noise
from .Random_Noise import Random_Noise


class DW_EncoderDecoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

#	def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder):
	def __init__(self, message_length, noise_layers_G, noise_layers_A, attention_encoder, attention_decoder):

		super(DW_EncoderDecoder, self).__init__()
		self.encoder = DW_Encoder(message_length, attention = attention_encoder)
		#self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
		self.noise = Random_Noise(noise_layers_G + noise_layers_A, len(noise_layers_G), len(noise_layers_A))

		#self.decoder_C = DW_Decoder(message_length, attention = attention_decoder)
		#self.decoder_RF = DW_Decoder(message_length, attention = attention_decoder)
		self.decoder_G = DW_Decoder(message_length, attention = attention_decoder)
		self.decoder_A = DW_Decoder(message_length, attention = attention_decoder)


	def forward(self, image, message, mask):
		encoded_image = self.encoder(image, message)
		#noised_image_C, noised_image_R, noised_image_F = self.noise([encoded_image, image, mask])
		noised_image_G, noised_image_A = self.noise([encoded_image, image, mask])
		#decoded_message_C = self.decoder_C(noised_image_C)
		#decoded_message_R = self.decoder_RF(noised_image_R)
		#decoded_message_F = self.decoder_RF(noised_image_F)
		decoded_message_G = self.decoder_G(noised_image_G)
		decoded_message_A = self.decoder_A(noised_image_A)
		#return encoded_image, noised_image_C, decoded_message_C, decoded_message_R, decoded_message_F
		return encoded_image, noised_image_G, noised_image_A, decoded_message_G, decoded_message_A

