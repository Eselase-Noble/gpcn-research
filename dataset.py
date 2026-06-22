"""
Dataset module for BreakHis with patient-level splitting
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from collections import defaultdict
import logging

from augmentation import get_train_transform, get_val_transform


logger = logging.getLogger(__name__)


class BREAKHISDataset(Dataset):
    """
    BreakHis Dataset with proper patient-level splitting
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 magnification = '40X',
                 transform: Optional[transforms.Compose] = None,
                 patient_ids: Optional[List[str]] = None,
                 use_patient_split: bool = True):
        """
        Args:
            root_dir: Root directory of BreakHis dataset
            split: 'train', 'val', or 'test'
            magnification: a single magnification ('40X'/'100X'/'200X'/'400X'),
                or a list of magnifications to POOL into one dataset (used for
                the magnification-agnostic 'all-magnifications' experiment).
            transform: Transformations to apply
            patient_ids: List of patient IDs for this split
            use_patient_split: Use patient-level splitting
        """
        self.root_dir = Path(root_dir)
        self.split = split
        # Normalise to a list of magnifications so pooling is just len>1.
        self.magnifications = [magnification] if isinstance(magnification, str) else list(magnification)
        self.magnification = magnification  # kept for backward-compat / logging
        self.transform = transform
        self.use_patient_split = use_patient_split
        
        # Class mapping
        self.classes = ['benign', 'malignant']
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        # Collect all images
        self.samples = []
        self._collect_samples(patient_ids)
        
        logger.info(f"{split} split: {len(self.samples)} images")
        self._print_class_distribution()

    def _extract_patient_id(self, img_path: Path) -> str:
        """Extract patient ID from image filename"""
        # Format: SOB_B_A-14-22549AB_40X_0001.png
        # Patient ID: 14-22549AB
        filename = img_path.stem
        parts = filename.split('_')

        try:
            patient_id = parts[2]  # Patient ID is 3rd part
            return patient_id
        except:
            return filename

    def _collect_samples(self, patient_ids: Optional[List[str]] = None):
        """Collect image samples across one or more magnifications."""
        # Pattern: histology_slides/breast/*/SOB/*/*/MAGNIFICATION/*.png
        for mag in self.magnifications:
            pattern = f"histology_slides/breast/*/SOB/*/*/{mag}/*.png"

            for img_path in self.root_dir.glob(pattern):
                # Extract class from path
                parts = img_path.parts
                try:
                    breast_idx = parts.index('breast')
                    class_name = parts[breast_idx + 1]  # benign or malignant

                    if class_name not in self.class_to_idx:
                        continue

                    # Extract patient ID
                    patient_id = self._extract_patient_id(img_path)

                    # Filter by patient IDs if using patient-level split
                    if self.use_patient_split and patient_ids is not None:
                        if patient_id not in patient_ids:
                            continue

                    # Add sample
                    self.samples.append({
                        'path': str(img_path),
                        'label': self.class_to_idx[class_name],
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'magnification': mag
                    })
                except (ValueError, IndexError):
                    continue

    def _print_class_distribution(self):
        """Print class distribution"""
        labels = [s['label'] for s in self.samples]
        benign_count = labels.count(0)
        malignant_count = labels.count(1)

        logger.info(f"  Benign: {benign_count} ({benign_count/len(labels)*100:.1f}%)")
        logger.info(f"  Malignant: {malignant_count} ({malignant_count/len(labels)*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Transformed image tensor
            label: Class label (0 or 1)
            patient_id: Patient identifier
        """
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample['path']).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, sample['label'], sample['patient_id']

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data"""
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels)
        class_weights = len(labels) / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)


def create_patient_level_split(root_dir: str,
                               magnification = '40X',
                               test_size: float = 0.2,
                               val_size: float = 0.1,
                               seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create patient-level train/val/test split

    magnification may be a single string or a list (pooled mode); patients are
    enumerated across all given magnifications so the same patient never lands in
    two splits regardless of magnification.

    Returns:
        train_patients, val_patients, test_patients
    """
    root = Path(root_dir)
    mags = [magnification] if isinstance(magnification, str) else list(magnification)

    # Collect all samples with patient IDs
    patient_to_label = {}
    for mag in mags:
        pattern = f"histology_slides/breast/*/SOB/*/*/{mag}/*.png"

        for img_path in root.glob(pattern):
            parts = img_path.parts
            try:
                breast_idx = parts.index('breast')
                class_name = parts[breast_idx + 1]

                if class_name not in ['benign', 'malignant']:
                    continue

                # Extract patient ID
                filename = img_path.stem
                patient_id = filename.split('_')[2]

                # Store patient label (use majority vote if patient has mixed labels)
                label = 0 if class_name == 'benign' else 1
                if patient_id not in patient_to_label:
                    patient_to_label[patient_id] = []
                patient_to_label[patient_id].append(label)
            except (ValueError, IndexError):
                continue

    # Get consensus label for each patient
    patients = []
    labels = []
    for patient_id, patient_labels in patient_to_label.items():
        patients.append(patient_id)
        # Use majority vote
        labels.append(1 if sum(patient_labels) > len(patient_labels) / 2 else 0)

    logger.info(f"\nPatient-level split:")
    logger.info(f"  Total patients: {len(patients)}")
    logger.info(f"  Benign patients: {labels.count(0)}")
    logger.info(f"  Malignant patients: {labels.count(1)}")

    # Split: train+val / test
    train_val_patients, test_patients, train_val_labels, test_labels = train_test_split(
        patients, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    # Split train into train / val
    val_size_adjusted = val_size / (1 - test_size)
    train_patients, val_patients, _, _ = train_test_split(
        train_val_patients, train_val_labels,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=train_val_labels
    )

    logger.info(f"  Train patients: {len(train_patients)}")
    logger.info(f"  Val patients: {len(val_patients)}")
    logger.info(f"  Test patients: {len(test_patients)}")

    return train_patients, val_patients, test_patients


def create_kfold_splits(root_dir: str,
                       magnification = '40X',
                       n_folds: int = 5,
                       seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    Create K-fold patient-level cross-validation splits.
    magnification may be a single string or a list (pooled mode).

    Returns:
        List of (train_patients, val_patients) tuples
    """
    root = Path(root_dir)
    mags = [magnification] if isinstance(magnification, str) else list(magnification)

    # Collect patient IDs and labels
    patient_to_label = {}
    for mag in mags:
        pattern = f"histology_slides/breast/*/SOB/*/*/{mag}/*.png"

        for img_path in root.glob(pattern):
            parts = img_path.parts
            try:
                breast_idx = parts.index('breast')
                class_name = parts[breast_idx + 1]

                if class_name not in ['benign', 'malignant']:
                    continue

                filename = img_path.stem
                patient_id = filename.split('_')[2]

                label = 0 if class_name == 'benign' else 1
                if patient_id not in patient_to_label:
                    patient_to_label[patient_id] = []
                patient_to_label[patient_id].append(label)
            except (ValueError, IndexError):
                continue

    # Get consensus labels
    patients = list(patient_to_label.keys())
    labels = [1 if sum(patient_to_label[p]) > len(patient_to_label[p]) / 2 else 0
              for p in patients]

    # Create stratified K-fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []
    for train_idx, val_idx in skf.split(patients, labels):
        train_patients = [patients[i] for i in train_idx]
        val_patients = [patients[i] for i in val_idx]
        splits.append((train_patients, val_patients))

    logger.info(f"\nCreated {n_folds}-fold cross-validation splits")

    return splits


def create_dataloaders(config, fold: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders

    Args:
        config: Experiment configuration
        fold: If provided, use K-fold split (for cross-validation)

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    # Get transforms
    train_transform = get_train_transform(config)
    val_transform = get_val_transform(config)

    # Magnification spec: pool ALL magnifications (magnification-agnostic model)
    # or use the single configured training magnification (per-mag protocol).
    if getattr(config.data, 'pool_all_magnifications', False):
        mag_spec = list(config.data.magnifications)
        logger.info(f"Magnification-POOLED mode: training on {mag_spec} combined")
    else:
        mag_spec = config.data.train_magnification

    if config.data.use_patient_split:
        if fold is not None:
            # K-fold cross-validation
            splits = create_kfold_splits(
                config.data.data_root,
                mag_spec,
                config.validation.n_folds,
                config.seed
            )
            train_patients, val_patients = splits[fold]
            test_patients = None  # Use validation as test in K-fold
        else:
            # Single train/val/test split
            train_patients, val_patients, test_patients = create_patient_level_split(
                config.data.data_root,
                mag_spec,
                config.data.test_size,
                config.data.val_size,
                config.seed
            )
    else:
        train_patients = val_patients = test_patients = None

    # Create datasets
    train_dataset = BREAKHISDataset(
        root_dir=config.data.data_root,
        split='train',
        magnification=mag_spec,
        transform=train_transform,
        patient_ids=train_patients,
        use_patient_split=config.data.use_patient_split
    )

    val_dataset = BREAKHISDataset(
        root_dir=config.data.data_root,
        split='val',
        magnification=mag_spec,
        transform=val_transform,
        patient_ids=val_patients,
        use_patient_split=config.data.use_patient_split
    )

    if test_patients is not None:
        test_dataset = BREAKHISDataset(
            root_dir=config.data.data_root,
            split='test',
            magnification=mag_spec,
            transform=val_transform,
            patient_ids=test_patients,
            use_patient_split=config.data.use_patient_split
        )
    else:
        test_dataset = val_dataset  # Use validation as test in K-fold

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True  # For MixUp/CutMix
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    # Get class weights
    class_weights = train_dataset.get_class_weights()

    # Create info dict
    info = {
        'num_classes': config.model.num_classes,
        'class_weights': class_weights,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'train_patients': len(train_patients) if train_patients else 'N/A',
        'val_patients': len(val_patients) if val_patients else 'N/A',
        'test_patients': len(test_patients) if test_patients else 'N/A',
    }

    logger.info(f"\nDataloader Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    return train_loader, val_loader, test_loader, info


def create_augmented_dataloaders(config, device):
    """
    Create dataloaders with optional GAN-based augmentation

    This wrapper integrates with the complete GAN augmentation system
    from gan_augmentation.py when available, with fallback to basic dataloaders.
    """
    # Try to use the complete GAN augmentation implementation
    try:
        from gan_augmentation import create_augmented_dataloaders as gan_augmented

        # Use full GAN system with statistics and proper generation
        logger.info("Using GAN augmentation system from gan_augmentation.py")
        return gan_augmented(
            config=config,
            device=device,
            generator_path=None  # Set to checkpoint path after training GAN
        )
    except ImportError:
        # Fallback: gan_augmentation.py not available
        logger.warning("gan_augmentation.py not found. Using basic dataloaders without GAN augmentation.")
        logger.warning("To enable GAN augmentation: ensure gan_augmentation.py is in the same directory.")
        return create_dataloaders(config)


def test_dataset():
    """Test dataset loading"""
    from config import get_default_config

    config = get_default_config()

    try:
        train_loader, val_loader, test_loader, info = create_dataloaders(config)

        print("\n✓ Dataset loading successful!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Test a batch
        images, labels, patient_ids = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Patient IDs: {len(patient_ids)}")

        return True
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_dataset()