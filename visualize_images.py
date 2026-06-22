"""
Visualize Original vs GAN-Generated Images
Shows side-by-side comparison of real histology images and synthetic images
"""

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from pathlib import Path
from PIL import Image

from config import get_default_config
from dataset import create_dataloaders
from gan_augmentation import Generator, AugmentedBREAKHISDataset


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor back to [0, 1] range for visualization
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_original_images(num_images=16, save_path='original_images.png'):
    """
    Display grid of original dataset images

    Args:
        num_images: Number of images to display
        save_path: Where to save the visualization
    """
    print("Loading original images from dataset...")

    config = get_default_config()
    train_loader, _, _, _ = create_dataloaders(config)

    # Get a batch of images
    images, labels, patient_ids = next(iter(train_loader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Denormalize for visualization
    images_denorm = denormalize(images)

    # Create figure
    fig = plt.figure(figsize=(16, 16))
    plt.suptitle('Original BreakHis Dataset Images', fontsize=20, fontweight='bold')

    for idx in range(num_images):
        ax = plt.subplot(4, 4, idx + 1)
        img = images_denorm[idx].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)

        label_name = 'Malignant' if labels[idx] == 1 else 'Benign'
        color = 'red' if labels[idx] == 1 else 'green'
        ax.set_title(label_name, color=color, fontweight='bold', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")
    plt.show()

    return images_denorm, labels


def visualize_gan_generated_images(generator_path, num_images=16, save_path='generated_images.png', device='cuda'):
    """
    Display grid of GAN-generated images

    Args:
        generator_path: Path to trained generator checkpoint
        num_images: Number of images to generate
        save_path: Where to save the visualization
        device: cuda or cpu
    """
    print(f"Loading generator from: {generator_path}")

    # Load generator
    generator = Generator(latent_dim=100, img_size=224).to(device)
    checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    print(f"Generating {num_images} synthetic images...")

    # Generate images
    with torch.no_grad():
        # Half benign, half malignant (just for visualization - labels don't affect generation)
        latent_vectors = torch.randn(num_images, 100).to(device)
        fake_images = generator(latent_vectors)

        # Rescale from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2

    # Create figure
    fig = plt.figure(figsize=(16, 16))
    plt.suptitle('GAN-Generated Synthetic Images', fontsize=20, fontweight='bold')

    for idx in range(num_images):
        ax = plt.subplot(4, 4, idx + 1)
        img = fake_images[idx].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f'Synthetic #{idx + 1}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")
    plt.show()

    return fake_images


def visualize_comparison(generator_path=None, num_pairs=8, save_path='comparison.png', device='cuda'):
    """
    Side-by-side comparison: Original (left) vs GAN-generated (right)

    Args:
        generator_path: Path to trained generator checkpoint (if None, generates random)
        num_pairs: Number of image pairs to show
        save_path: Where to save the visualization
        device: cuda or cpu
    """
    print("\n" + "=" * 70)
    print("CREATING ORIGINAL vs GAN-GENERATED COMPARISON")
    print("=" * 70)

    # Get original images
    print("\n1. Loading original images...")
    config = get_default_config()
    train_loader, _, _, _ = create_dataloaders(config)
    real_images, real_labels, _ = next(iter(train_loader))
    real_images = real_images[:num_pairs]
    real_labels = real_labels[:num_pairs]
    real_images_denorm = denormalize(real_images)

    # Generate synthetic images
    print("2. Generating synthetic images...")
    if generator_path and Path(generator_path).exists():
        generator = Generator(latent_dim=100, img_size=224).to(device)
        checkpoint = torch.load(generator_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        generator.eval()

        with torch.no_grad():
            z = torch.randn(num_pairs, 100).to(device)
            fake_images = generator(z)
            fake_images = (fake_images + 1) / 2  # Rescale to [0, 1]
    else:
        print("   (No generator provided - using placeholder)")
        fake_images = torch.rand(num_pairs, 3, 224, 224)

    # Create comparison figure
    print("3. Creating comparison visualization...")
    fig = plt.figure(figsize=(18, num_pairs * 2.5))
    plt.suptitle('Original (Real) vs GAN-Generated (Synthetic) Images',
                 fontsize=22, fontweight='bold', y=0.995)

    for idx in range(num_pairs):
        # Original image
        ax1 = plt.subplot(num_pairs, 2, idx * 2 + 1)
        real_img = real_images_denorm[idx].permute(1, 2, 0).cpu().numpy()
        ax1.imshow(real_img)

        label_name = 'Malignant' if real_labels[idx] == 1 else 'Benign'
        color = 'red' if real_labels[idx] == 1 else 'green'
        ax1.set_title(f'ORIGINAL - {label_name}', color=color, fontweight='bold', fontsize=14)
        ax1.axis('off')
        ax1.text(0.5, -0.05, 'Real Histology Image', ha='center', va='top',
                 transform=ax1.transAxes, fontsize=10, style='italic', color='blue')

        # Generated image
        ax2 = plt.subplot(num_pairs, 2, idx * 2 + 2)
        fake_img = fake_images[idx].permute(1, 2, 0).cpu().numpy()
        ax2.imshow(fake_img)
        ax2.set_title(f'GAN-GENERATED #{idx + 1}', color='purple', fontweight='bold', fontsize=14)
        ax2.axis('off')
        ax2.text(0.5, -0.05, 'Synthetic Image', ha='center', va='top',
                 transform=ax2.transAxes, fontsize=10, style='italic', color='purple')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {save_path}")
    print("=" * 70 + "\n")
    plt.show()


def visualize_augmented_dataset_samples(generator_path, num_samples=16, save_path='augmented_samples.png',
                                        device='cuda'):
    """
    Show samples from the full augmented dataset (mix of real + synthetic)

    Args:
        generator_path: Path to trained generator
        num_samples: Number of samples to show
        save_path: Where to save
        device: cuda or cpu
    """
    print("\n" + "=" * 70)
    print("VISUALIZING AUGMENTED DATASET (Real + Synthetic Mix)")
    print("=" * 70)

    # Create augmented dataset
    print("\n1. Creating augmented dataset...")
    config = get_default_config()
    train_loader, _, _, _ = create_dataloaders(config)
    base_dataset = train_loader.dataset

    # Load generator
    generator = Generator(latent_dim=100, img_size=224).to(device)
    checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])

    # Create augmented dataset
    augmented_dataset = AugmentedBREAKHISDataset(
        base_dataset=base_dataset,
        generator=generator,
        num_synthetic_per_class={0: 4, 1: 4},  # Generate 4 of each for visualization
        device=device,
        apply_traditional_aug=False  # Disable for clearer visualization
    )

    print("2. Sampling from augmented dataset...")

    # Sample images
    fig = plt.figure(figsize=(16, 16))
    plt.suptitle('Augmented Dataset: Mix of Original + GAN-Generated',
                 fontsize=20, fontweight='bold')

    for idx in range(min(num_samples, len(augmented_dataset))):
        img, label, patient_id = augmented_dataset[idx]

        ax = plt.subplot(4, 4, idx + 1)

        # Denormalize if needed
        if img.min() < 0:  # If normalized
            img_denorm = denormalize(img.unsqueeze(0)).squeeze(0)
        else:
            img_denorm = img

        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)

        # Label
        label_name = 'Malignant' if label == 1 else 'Benign'
        color = 'red' if label == 1 else 'green'

        # Check if synthetic
        is_synthetic = patient_id == 'synthetic'
        title = f"{'SYNTHETIC' if is_synthetic else 'ORIGINAL'} - {label_name}"
        title_color = 'purple' if is_synthetic else color

        ax.set_title(title, color=title_color, fontweight='bold', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {save_path}")
    print("=" * 70 + "\n")
    plt.show()


def quick_visual_check(generator_path=None):
    """
    Quick function to check if images look realistic
    Shows 4 original and 4 generated side-by-side
    """
    print("\n" + "=" * 70)
    print("QUICK VISUAL CHECK: Do GAN images look realistic?")
    print("=" * 70 + "\n")

    if generator_path is None:
        print("⚠ No generator path provided. Please provide a trained generator checkpoint.")
        print("Example: quick_visual_check('gan_checkpoints/checkpoint_epoch_50.pth')")
        return

    visualize_comparison(
        generator_path=generator_path,
        num_pairs=4,
        save_path='quick_check.png',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n✓ Check the image above:")
    print("  - Do synthetic images look like histology slides?")
    print("  - Are cell structures visible?")
    print("  - Are colors realistic (purple/pink staining)?")
    print("\nIf YES → GAN is working well, use for training")
    print("If NO  → Train GAN longer or check training data")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS FOR JUPYTER NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════════

def show_dataset(num_images=16):
    """Quick function to display dataset images"""
    return visualize_original_images(num_images=num_images, save_path='dataset_preview.png')


def show_generated(checkpoint_path, num_images=16):
    """Quick function to display GAN-generated images"""
    return visualize_gan_generated_images(
        generator_path=checkpoint_path,
        num_images=num_images,
        save_path='gan_preview.png'
    )


def compare(checkpoint_path=None, num_pairs=8):
    """Quick function to compare original vs generated"""
    visualize_comparison(
        generator_path=checkpoint_path,
        num_pairs=num_pairs,
        save_path='comparison.png'
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Original vs GAN-Generated Images')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['original', 'generated', 'comparison', 'augmented', 'quick'],
                        help='Visualization mode')
    parser.add_argument('--generator', type=str, default=None,
                        help='Path to trained generator checkpoint')
    parser.add_argument('--num-images', type=int, default=16,
                        help='Number of images to display')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print("⚠ CUDA not available, using CPU")

    # Execute based on mode
    if args.mode == 'original':
        output = args.output or 'original_images.png'
        visualize_original_images(num_images=args.num_images, save_path=output)

    elif args.mode == 'generated':
        if args.generator is None:
            print("❌ Error: --generator path required for 'generated' mode")
            print(
                "Example: python visualize_images.py --mode generated --generator gan_checkpoints/checkpoint_epoch_50.pth")
        else:
            output = args.output or 'generated_images.png'
            visualize_gan_generated_images(
                generator_path=args.generator,
                num_images=args.num_images,
                save_path=output,
                device=device
            )

    elif args.mode == 'comparison':
        output = args.output or 'comparison.png'
        visualize_comparison(
            generator_path=args.generator,
            num_pairs=args.num_images // 2,
            save_path=output,
            device=device
        )

    elif args.mode == 'augmented':
        if args.generator is None:
            print("❌ Error: --generator path required for 'augmented' mode")
        else:
            output = args.output or 'augmented_samples.png'
            visualize_augmented_dataset_samples(
                generator_path=args.generator,
                num_samples=args.num_images,
                save_path=output,
                device=device
            )

    elif args.mode == 'quick':
        quick_visual_check(generator_path=args.generator)

    print("\n✓ Visualization complete!")