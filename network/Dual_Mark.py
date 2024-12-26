from .DW_EncoderDecoder import *
from .Patch_Discriminator import Patch_Discriminator
import torch
import kornia.losses
import lpips


class Network:
	def __init__(self, message_length, noise_layers_G, noise_layers_A, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight):
		# device
		self.device = device

		# loss function
		self.criterion_MSE = nn.MSELoss().to(device)
		self.criterion_LPIPS = lpips.LPIPS().to(device)

		# weight of encoder-decoder loss
		self.encoder_weight = weight[0]
		self.decoder_weight_G = weight[1]
		self.decoder_weight_A = weight[2]
		self.detector_weight = weight[3]
		self.discriminator_weight = weight[4]

		# network
		self.encoder_decoder = DW_EncoderDecoder(message_length, noise_layers_G, noise_layers_A, attention_encoder, attention_decoder).to(device)		
		self.discriminator = Patch_Discriminator().to(device)

		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
		self.discriminator = torch.nn.DataParallel(self.discriminator)

		# mark "cover" as 1, "encoded" as -1
		self.label_cover = 1.0
		self.label_encoded = - 1.0
		for p in self.encoder_decoder.module.noise.parameters():
			p.requires_grad = False

		# optimizer
		self.opt_encoder_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr, betas=(beta1, 0.999))
		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))


	def train(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor):
		self.encoder_decoder.train()
		self.discriminator.train()

		with torch.enable_grad():
			# use device to compute
			images, messages, masks = images.to(self.device), messages.to(self.device), masks.to(self.device)
			encoded_images, noised_images_G, noised_images_A, decoded_messages_G, decoded_messages_A, decoded_messages_G_I, decoded_messages_A_I = self.encoder_decoder(images, messages, masks)

			'''
			train discriminator
			'''
			for p in self.discriminator.parameters():
				p.requires_grad = True

			self.opt_discriminator.zero_grad()

			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
   
			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())

			d_loss = self.criterion_MSE(d_label_cover - torch.mean(d_label_encoded), self.label_cover * torch.ones_like(d_label_cover)) +\
			         self.criterion_MSE(d_label_encoded - torch.mean(d_label_cover), self.label_encoded * torch.ones_like(d_label_encoded))
			d_loss.backward()

			self.opt_discriminator.step()

			'''
			train encoder and decoder
			'''
			# Make it a tiny bit faster
			for p in self.discriminator.parameters():
				p.requires_grad = False

			self.opt_encoder_decoder.zero_grad()

			# GAN : target label for encoded image should be "cover"(0)
			g_label_cover = self.discriminator(images)
			g_label_encoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_MSE(g_label_cover - torch.mean(g_label_encoded), self.label_encoded * torch.ones_like(g_label_cover)) +\
									  self.criterion_MSE(g_label_encoded - torch.mean(g_label_cover), self.label_cover * torch.ones_like(g_label_encoded))

			# RAW : the encoded image should be similar to cover image
			g_loss_on_encoder_MSE = self.criterion_MSE(encoded_images, images)
			g_loss_on_encoder_LPIPS = torch.mean(self.criterion_LPIPS(encoded_images, images))

			# RESULT : the decoded message should be similar to the raw message /Dual
			g_loss_on_decoder_G = self.criterion_MSE(decoded_messages_G, messages)
			g_loss_on_decoder_A = self.criterion_MSE(decoded_messages_A, messages)


			# DETECT : decoders should not detect distortion besides deepfake
			g_loss_on_decoder_G_I = self.criterion_MSE(decoded_messages_G_I, torch.zeros_like(messages))
			g_loss_on_decoder_A_I = self.criterion_MSE(decoded_messages_A_I, torch.zeros_like(messages))


			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder_MSE +\
					 self.decoder_weight_G * g_loss_on_decoder_G + self.decoder_weight_A * g_loss_on_decoder_A +\
					 self.detector_weight * g_loss_on_decoder_G_I + self.detector_weight * g_loss_on_decoder_A_I
			
			g_loss.backward()
			self.opt_encoder_decoder.step()

			# psnr
			psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean")

		'''
		decoded message error rate /Dual
		'''
		error_rate_G = self.decoded_message_error_rate_batch(messages, decoded_messages_G)
		error_rate_A = self.decoded_message_error_rate_batch(messages, decoded_messages_A)

		error_rate_G_I = self.decoded_message_error_rate_batch(messages, decoded_messages_G_I)
		error_rate_A_I = self.decoded_message_error_rate_batch(messages, decoded_messages_A_I)

		result = {
			"g_loss": g_loss,
			"error_rate_G": error_rate_G,
			"error_rate_A": error_rate_A,
			"error_rate_G_I": error_rate_G_I,
			"error_rate_A_I": error_rate_A_I,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder_MSE": g_loss_on_encoder_MSE,
			"g_loss_on_encoder_LPIPS": g_loss_on_encoder_LPIPS,
			"g_loss_on_decoder_G": g_loss_on_decoder_G,
			"g_loss_on_decoder_A": g_loss_on_decoder_A,
			"g_loss_on_decoder_G_I" : g_loss_on_decoder_G_I,
			"g_loss_on_decoder_A_I" : g_loss_on_decoder_A_I,
			"d_loss": d_loss
		}
		return result, (images, encoded_images, noised_images_G, noised_images_A)

	def validation(self, images: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor, noise_layer):
		self.encoder_decoder.eval()
		self.discriminator.eval()
  
		with torch.no_grad():
			images, messages, masks = images.to(self.device), messages.to(self.device), masks.to(self.device)
			encoded_images = self.encoder_decoder.module.encoder(images, messages)
			noised_images = noise_layer([encoded_images.clone(), images.clone(), masks.clone()])

			decoded_messages_G = self.encoder_decoder.module.decoder_G(noised_images)
			decoded_messages_A = self.encoder_decoder.module.decoder_A(noised_images)
   
			decoded_messages_G_I = self.encoder_decoder.module.decoder_G(encoded_images.clone())
			decoded_messages_A_I = self.encoder_decoder.module.decoder_A(encoded_images.clone())
   
			error_rate_G = self.decoded_message_error_rate_batch(messages, decoded_messages_G)
			error_rate_A = self.decoded_message_error_rate_batch(messages, decoded_messages_A)
			error_rate_G_I = self.decoded_message_error_rate_batch(messages, decoded_messages_G_I)
			error_rate_A_I = self.decoded_message_error_rate_batch(messages, decoded_messages_A_I)

			result = {
				"error_rate_G": error_rate_G,
				"error_rate_A": error_rate_A,
				"error_rate_G_I": error_rate_G_I,
				"error_rate_A_I": error_rate_A_I
			}
   
			decoded_messages = {
				"decoded_messages_G": decoded_messages_G,
				"decoded_messages_A": decoded_messages_A,
				"decoded_messages_G_I": decoded_messages_G_I,
				"decoded_messages_A_I": decoded_messages_A_I
			}

			return result, images, encoded_images, noised_images, decoded_messages

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0)
		decoded_message = decoded_message.gt(0)
		error_rate = float(sum(message != decoded_message)) / length
		return error_rate

	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
		error_rate /= batch_size
		return error_rate

	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.module.state_dict(), path_discriminator)

	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)

	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

	def load_model_dis(self, path_discriminator: str):
		self.discriminator.module.load_state_dict(torch.load(path_discriminator))
