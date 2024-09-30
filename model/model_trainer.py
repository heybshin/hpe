import torch
import torch.nn as nn
import logging

from braindecode.models import EEGNetv4
from pretrained.detr.models import build_model
from pretrained.DistilHuBERT.DistilHuBERT import build_distilhubert
from pretrained.detr.models.detr import MLP
from model.model_utils import get_eeg_embedding_hook, get_aud_embedding_hook, get_tou_embedding_hook

class ModelTrainer:
    def __init__(self, args, config):
        self.args = args
        self.device = args.device
        self.config = config

        # Load models
        self.load_pretrained_models()

        # Initialize loss functions and optimizer
        self.init_training_components()

        # Save initial model states
        self.save_model_states()

    def load_pretrained_models(self):
        # Load DETR model
        self.detr, _, _ = build_model(self.args)
        self.detr.to(self.device)
        detr_checkpoint = torch.load(self.config['model_paths']['detr'], map_location='cpu')
        self.detr.load_state_dict(detr_checkpoint['model'], strict=False)

        # Load EEGNet model
        self.eegnet = EEGNetv4(n_chans=1, n_outputs=2, n_times=750, final_conv_length='auto')
        self.eegnet.to(self.device)
        eegnet_path = self.config['model_paths']['eegnet'].format(subject=self.args.subject)
        self.eegnet.load_state_dict(torch.load(eegnet_path, map_location='cpu'))

        # Load MLP model
        self.mlp = MLP(2, 368, 8, 3)
        self.mlp.to(self.device)
        mlp_path = self.config['model_paths']['mlp']
        self.mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))

        # Load DistilHuBERT model
        distilhubert_ckpt = self.config['model_paths']['distilhubert']
        self.distilhubert = build_distilhubert(distilhubert_ckpt).to(self.device)

        # Freeze or unfreeze model parameters as needed
        self.freeze_params()

    def freeze_params(self):
        # Freeze EEGNet parameters
        for param in self.eegnet.parameters():
            param.requires_grad = False

        # Freeze DistilHuBERT parameters
        for param in self.distilhubert.parameters():
            param.requires_grad = False

        # Freeze MLP parameters
        for param in self.mlp.parameters():
            param.requires_grad = False

        # Freeze DETR decoder
        for param in self.detr.transformer.decoder.parameters():
            param.requires_grad = False

        # Unfreeze DETR backbone and encoder
        for param in self.detr.backbone.parameters():
            param.requires_grad = True
        for param in self.detr.transformer.encoder.parameters():
            param.requires_grad = True

    def init_training_components(self):
        # Initialize alignment network
        self.alignNet = AlignNet().to(self.device)
        self.feature_fuser = FeatureFusion(368).to(self.device)

        # Extract classification head from EEGNet
        self.classification_head = nn.Sequential()
        for idx, (name, module) in enumerate(self.eegnet.named_children()):
            if idx > 13:
                self.classification_head.add_module(name, module)
        self.classification_head.to(self.device)

        # Define loss functions
        self.l_align = nn.L1Loss()
        self.l_cls = nn.CrossEntropyLoss()

        # Initialize optimizer
        params = list(self.alignNet.parameters()) + \
                 list(self.detr.backbone.parameters()) + \
                 list(self.detr.transformer.encoder.parameters()) + \
                 list(self.feature_fuser.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config['training_params']['learning_rate'])

    def save_model_states(self):
        # Save initial model states
        torch.save(self.detr.backbone.state_dict(), self.args.baselog_dir + '/detr_backbone.pth')
        torch.save(self.detr.transformer.encoder.state_dict(), self.args.baselog_dir + '/detr_transformer_encoder.pth')
        torch.save(self.feature_fuser.state_dict(), self.args.baselog_dir + '/feature_fuser.pth')
        torch.save(self.alignNet.state_dict(), self.args.baselog_dir + '/alignNet.pth')

    def train(self, training_data):
        # Unpack training data
        trial_counter, input_eeg, target_eeg, image, audio, touch, bo_params = training_data
        # Train the model and get the target value for BO
        target = self.train_one_iteration(trial_counter, input_eeg, target_eeg, image, audio, touch)
        return target

    def train_one_iteration(self, trial, input_eeg, target_eeg, image, audio, touch):
        # Implement your EEG data processing here
        logging.info("Processing EEG data...")

        # Get the hook function and handle
        get_eeg_output, hook_handle_eeg = get_eeg_embedding_hook(self.eegnet)
        get_aud_output, hook_handle_aud = get_aud_embedding_hook(self.distilhubert)
        get_tou_output, hook_handle_tou = get_tou_embedding_hook(self.mlp)

        # Perform a forward pass with your input data
        input_eeg = torch.tensor(input_eeg, dtype=torch.float32, device=self.device)
        target_eeg = torch.tensor(target_eeg, dtype=torch.long, device=self.device)

        # Forward pass through models
        output_detr, out_encoder = self.detr(image.to(self.device), hid_out=True)
        output_aud = self.distilhubert(audio['train']['input_values'].to(self.device))
        output_tou = self.mlp(touch.to(self.device))
        output_eeg = self.eegnet(input_eeg)

        # Get embeddings
        image_embed = self.alignNet(out_encoder)
        audio_embed = get_aud_output().mean(dim=1).view(1, image_embed.shape[1], image_embed.shape[2], image_embed.shape[3])
        touch_embed = get_tou_output()
        brain_embed = get_eeg_output()

        # Fuse embeddings
        trans_embed = self.feature_fuser(image_embed, audio_embed, touch_embed)

        # Compute losses
        self.optimizer.zero_grad()
        loss = self.l_cls(self.classification_head(trans_embed), target_eeg)
        loss += self.l_align(brain_embed, trans_embed) + self.l_align(self.classification_head(brain_embed), output_eeg)
        logging.info(f"Trial {trial} Loss: {loss.item():.4f}")
        with open(f'{self.args.baselog_dir}/loss.txt', 'a') as f:
            f.write(f'Trial_{trial}_Loss_{loss.item()}\n')

        # Backpropagation and optimization step
        loss.backward()
        self.optimizer.step()

        # Remove hooks
        hook_handle_eeg.remove()
        hook_handle_aud.remove()
        hook_handle_tou.remove()

        logging.info("="*40)

        # Return the negative loss as the target for Bayesian Optimization
        return -loss.item()

# Define AlignNet and FeatureFusion classes as per your original code
class AlignNet(nn.Module):
    def __init__(self):
        super(AlignNet, self).__init__()
        # Define layers
        self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d((16, 23))

    def forward(self, x):
        batch_size, d1, d2, d3 = x.shape
        x = self.conv1x1(x)
        x = x.view(batch_size, 1, 16, d2 * d3)
        x = self.pooling(x)
        x = x.view(batch_size, 16, 1, -1)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, embedding_dim):
        super(FeatureFusion, self).__init__()
        self.mask1 = nn.Parameter(torch.ones(embedding_dim))
        self.mask2 = nn.Parameter(torch.ones(embedding_dim))
        self.mask3 = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, embedding1, embedding2, embedding3):
        batch_size, d1, d2, d3 = embedding1.shape
        embedding1 = embedding1.view(batch_size, -1)
        embedding2 = embedding2.view(batch_size, -1)
        embedding3 = embedding3.view(batch_size, -1)

        weighted_embedding1 = embedding1 * self.mask1
        weighted_embedding2 = embedding2 * self.mask2
        weighted_embedding3 = embedding3 * self.mask3

        fused_embedding = weighted_embedding1 + weighted_embedding2 + weighted_embedding3
        fused_embedding = fused_embedding.view(batch_size, d1, d2, d3)
        return fused_embedding
