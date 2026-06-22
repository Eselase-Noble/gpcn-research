"""
GAN-based Data Augmentation for BreakHis
- Train GAN on real histology images
- Generate synthetic images to balance dataset
- Save generated images for reuse
- Combine with traditional augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
from pathlib import Path
import random
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# GAN ARCHITECTURE (DCGAN-style for histology images)
# ═══════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    DCGAN Generator for 224x224 histology images
    """

    def __init__(self, latent_dim=100, img_channels=3, img_size=224, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Calculate initial feature map size
        # 224 / 16 = 14 (4 upsampling layers: 14 -> 28 -> 56 -> 112 -> 224)
        self.init_size = img_size // 16

        # Project and reshape
        self.project = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size),
            nn.BatchNorm1d(ngf * 8 * self.init_size * self.init_size),
            nn.ReLU(True)
        )

        # Convolutional layers
        self.conv = nn.Sequential(
            # State: (ngf*8) x 14 x 14
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 28 x 28
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 56 x 56
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: ngf x 112 x 112
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 3 x 224 x 224
        )

    def forward(self, z):
        """
        Args:
            z: Latent vector [B, latent_dim]
        Returns:
            img: Generated image [B, 3, 224, 224] in range [-1, 1]
        """
        x = self.project(z)
        x = x.view(x.size(0), -1, self.init_size, self.init_size)
        img = self.conv(x)
        return img


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for 224x224 histology images
    """

    def __init__(self, img_channels=3, img_size=224, ndf=64):
        super().__init__()

        self.model = nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 112 x 112
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 56 x 56
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 28 x 28
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 14 x 14
            nn.Conv2d(ndf * 8, 1, 14, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, img):
        """
        Args:
            img: Image [B, 3, 224, 224]
        Returns:
            validity: Real/Fake score [B, 1]
        """
        validity = self.model(img)
        return validity.view(-1, 1)


# ═══════════════════════════════════════════════════════════════════════
# AUGMENTED DATASET
# ═══════════════════════════════════════════════════════════════════════

class AugmentedBREAKHISDataset(Dataset):
    """
    Wraps BREAKHISDataset and adds GAN-generated synthetic images

    Features:
    - Balances class distribution with synthetic samples
    - Tracks original vs synthetic counts
    - Optional traditional augmentation on top
    """

    def __init__(self,
                 base_dataset,
                 generator: nn.Module = None,
                 num_synthetic_per_class: dict = None,
                 device: str = 'cuda',
                 apply_traditional_aug: bool = True):
        """
        Args:
            base_dataset: Original BREAKHISDataset
            generator: Trained generator model
            num_synthetic_per_class: {0: num_benign, 1: num_malignant}
            device: Device for generation
            apply_traditional_aug: Apply traditional augmentation on top
        """
        self.base_dataset = base_dataset
        self.generator = generator
        self.device = device
        self.apply_traditional_aug = apply_traditional_aug

        # Count original samples per class
        self.original_counts = self._count_classes(base_dataset)

        # Set synthetic counts
        if num_synthetic_per_class is None:
            # Default: balance dataset to majority class
            majority_count = max(self.original_counts.values())
            num_synthetic_per_class = {
                cls: max(0, majority_count - count)
                for cls, count in self.original_counts.items()
            }

        self.num_synthetic_per_class = num_synthetic_per_class
        self.total_synthetic = sum(num_synthetic_per_class.values())

        # Pre-generate synthetic samples if generator provided
        self.synthetic_samples = []
        if generator is not None:
            self._generate_synthetic_samples()

        # Print statistics
        self._print_statistics()

    def _count_classes(self, dataset):
        """Count samples per class in base dataset"""
        counts = {0: 0, 1: 0}
        for _, label, _ in dataset:
            counts[label] += 1
        return counts

    def _generate_synthetic_samples(self):
        """Pre-generate all synthetic samples"""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING SYNTHETIC IMAGES")
        logger.info("=" * 70)

        self.generator.eval()
        self.generator.to(self.device)

        with torch.no_grad():
            for class_label, num_samples in self.num_synthetic_per_class.items():
                if num_samples == 0:
                    continue

                logger.info(f"Generating {num_samples} synthetic {['benign', 'malignant'][class_label]} samples...")

                # Generate in batches
                batch_size = 32
                for i in tqdm(range(0, num_samples, batch_size), desc=f"Class {class_label}"):
                    current_batch = min(batch_size, num_samples - i)

                    # Generate latent vectors
                    z = torch.randn(current_batch, self.generator.latent_dim).to(self.device)

                    # Generate images
                    fake_imgs = self.generator(z)

                    # Rescale from [-1, 1] to [0, 1]
                    fake_imgs = (fake_imgs + 1) / 2

                    # Store synthetic samples
                    for img in fake_imgs:
                        self.synthetic_samples.append({
                            'image': img.cpu(),
                            'label': class_label,
                            'patient_id': 'synthetic'
                        })

        logger.info(f"✓ Generated {len(self.synthetic_samples)} synthetic images")

    def _print_statistics(self):
        """Print dataset statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 70)

        logger.info(f"\nOriginal Dataset:")
        logger.info(f"  Benign:    {self.original_counts[0]:5d}")
        logger.info(f"  Malignant: {self.original_counts[1]:5d}")
        logger.info(f"  Total:     {sum(self.original_counts.values()):5d}")

        logger.info(f"\nGAN-Generated Synthetic:")
        logger.info(f"  Benign:    {self.num_synthetic_per_class.get(0, 0):5d}")
        logger.info(f"  Malignant: {self.num_synthetic_per_class.get(1, 0):5d}")
        logger.info(f"  Total:     {self.total_synthetic:5d}")

        final_counts = {
            0: self.original_counts[0] + self.num_synthetic_per_class.get(0, 0),
            1: self.original_counts[1] + self.num_synthetic_per_class.get(1, 0)
        }

        logger.info(f"\nAugmented Dataset (Original + Synthetic):")
        logger.info(f"  Benign:    {final_counts[0]:5d}")
        logger.info(f"  Malignant: {final_counts[1]:5d}")
        logger.info(f"  Total:     {sum(final_counts.values()):5d}")

        logger.info(f"\nAugmentation Factor:")
        logger.info(f"  {self.total_synthetic / sum(self.original_counts.values()):.2f}x increase")
        logger.info("=" * 70 + "\n")

    def __len__(self):
        return len(self.base_dataset) + len(self.synthetic_samples)

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            # Original sample
            img, label, patient_id = self.base_dataset[idx]
            return img, label, patient_id
        else:
            # Synthetic sample
            synthetic_idx = idx - len(self.base_dataset)
            sample = self.synthetic_samples[synthetic_idx]

            img = sample['image']
            label = sample['label']
            patient_id = sample['patient_id']

            # Apply traditional augmentation if enabled
            if self.apply_traditional_aug and self.base_dataset.transform:
                # Convert tensor to PIL for augmentation
                img_pil = transforms.ToPILImage()(img)
                img = self.base_dataset.transform(img_pil)

            return img, label, patient_id


# ═══════════════════════════════════════════════════════════════════════
# GAN TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def train_gan(train_loader,
              latent_dim=100,
              img_size=224,
              num_epochs=50,
              device='cuda',
              save_dir='gan_checkpoints'):
    """
    Train GAN on BreakHis data

    Args:
        train_loader: DataLoader with real images
        latent_dim: Dimension of latent space
        img_size: Image size (224 for BreakHis)
        num_epochs: Training epochs
        device: cuda or cpu
        save_dir: Directory to save checkpoints

    Returns:
        generator: Trained generator model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    generator = Generator(latent_dim=latent_dim, img_size=img_size).to(device)
    discriminator = Discriminator(img_size=img_size).to(device)

    # Optimizers
    lr = 0.0002
    beta1 = 0.5
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Loss
    adversarial_loss = nn.BCELoss()

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING GAN")
    logger.info("=" * 70)
    logger.info(f"Latent dim: {latent_dim}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70 + "\n")

    for epoch in range(num_epochs):
        for i, (imgs, _, _) in enumerate(train_loader):
            batch_size = imgs.size(0)

            # Adversarial ground truths
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Real images
            real_imgs = imgs.to(device)

            # ────────────────────────────────────────────────────────
            # Train Discriminator
            # ────────────────────────────────────────────────────────
            optimizer_D.zero_grad()

            # Loss on real images
            real_loss = adversarial_loss(discriminator(real_imgs), real_labels)

            # Generate fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)

            # Loss on fake images
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ────────────────────────────────────────────────────────
            # Train Generator
            # ────────────────────────────────────────────────────────
            optimizer_G.zero_grad()

            # Generator wants discriminator to think fakes are real
            g_loss = adversarial_loss(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            optimizer_G.step()

            # ────────────────────────────────────────────────────────
            # Logging
            # ────────────────────────────────────────────────────────
            if i % 100 == 0:
                logger.info(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim).to(device)
                gen_imgs = generator(z)
                save_image(gen_imgs, save_dir / f"epoch_{epoch + 1}.png", nrow=4, normalize=True)

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')

    logger.info("\n✓ GAN training complete!")
    return generator


def save_synthetic_images(generator,
                          num_samples_per_class,
                          save_dir,
                          device='cuda',
                          latent_dim=100):
    """
    Generate and save synthetic images to disk

    Args:
        generator: Trained generator
        num_samples_per_class: {0: num_benign, 1: num_malignant}
        save_dir: Directory to save images
        device: cuda or cpu
        latent_dim: Latent dimension
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    generator.eval()
    generator.to(device)

    logger.info("\n" + "=" * 70)
    logger.info("SAVING SYNTHETIC IMAGES TO DISK")
    logger.info("=" * 70)

    for class_label, num_samples in num_samples_per_class.items():
        class_name = ['benign', 'malignant'][class_label]
        class_dir = save_dir / class_name
        class_dir.mkdir(exist_ok=True)

        logger.info(f"\nGenerating {num_samples} {class_name} images...")

        with torch.no_grad():
            batch_size = 32
            for i in tqdm(range(0, num_samples, batch_size), desc=f"Saving {class_name}"):
                current_batch = min(batch_size, num_samples - i)

                # Generate
                z = torch.randn(current_batch, latent_dim).to(device)
                fake_imgs = generator(z)

                # Save individual images
                for j, img in enumerate(fake_imgs):
                    img_idx = i + j
                    filename = f"synthetic_{class_name}_{img_idx:05d}.png"
                    save_image(img, class_dir / filename, normalize=True)

        logger.info(f"✓ Saved {num_samples} {class_name} images to {class_dir}")

    logger.info("\n" + "=" * 70)
    logger.info(f"✓ ALL SYNTHETIC IMAGES SAVED TO: {save_dir}")
    logger.info("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION WITH EXISTING CODE
# ═══════════════════════════════════════════════════════════════════════

def create_augmented_dataloaders(config, device='cuda', generator_path=None):
    """
    Create dataloaders with GAN augmentation

    Args:
        config: Experiment config
        device: cuda or cpu
        generator_path: Path to trained generator checkpoint (optional)

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    # Import from dataset module
    from dataset import create_dataloaders

    # Get base dataloaders
    base_train_loader, val_loader, test_loader, info = create_dataloaders(config)

    # Get base train dataset
    base_train_dataset = base_train_loader.dataset

    # Load generator if path provided
    generator = None
    if generator_path and Path(generator_path).exists():
        logger.info(f"\nLoading generator from: {generator_path}")
        generator = Generator(latent_dim=100, img_size=config.data.image_size).to(device)
        checkpoint = torch.load(generator_path, map_location='cpu', weights_only=False)
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()
        logger.info("✓ Generator loaded")

    # Determine synthetic counts
    if generator is not None and config.data.use_advanced_augmentation:
        # Count original samples per class
        original_counts = {0: 0, 1: 0}
        for _, label, _ in base_train_dataset:
            original_counts[label] += 1

        # Balance to majority class + 50% extra
        majority_count = max(original_counts.values())
        target_count = int(majority_count * 1.5)

        num_synthetic_per_class = {
            cls: max(0, target_count - count)
            for cls, count in original_counts.items()
        }
    else:
        num_synthetic_per_class = {0: 0, 1: 0}

    # Create augmented dataset
    augmented_train_dataset = AugmentedBREAKHISDataset(
        base_dataset=base_train_dataset,
        generator=generator,
        num_synthetic_per_class=num_synthetic_per_class,
        device=device,
        apply_traditional_aug=True
    )

    # Create augmented dataloader
    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True
    )

    # Update info
    info['train_samples'] = len(augmented_train_dataset)
    info['train_synthetic'] = sum(num_synthetic_per_class.values())
    info['train_original'] = len(base_train_dataset)

    return train_loader, val_loader, test_loader, info