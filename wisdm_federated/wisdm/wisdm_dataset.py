"""WISDM Dataset download, preprocessing and partitioning utilities."""

import os
import zipfile
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


# Dataset URL
WISDM_URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
DATA_DIR = "./data/wisdm"
RAW_DATA_FILE = "WISDM_ar_v1.1_raw.txt"


def download_wisdm():
    """Download the WISDM dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    tar_path = os.path.join(DATA_DIR, "WISDM_ar_latest.tar.gz")
    
    if not os.path.exists(tar_path):
        print("Downloading WISDM dataset...")
        urlretrieve(WISDM_URL, tar_path)
        print("Download completed!")
    else:
        print("Dataset already downloaded.")
    
    # Extract the tar.gz file
    import tarfile
    extracted_dir = os.path.join(DATA_DIR, "WISDM_ar_v1.1")
    if not os.path.exists(extracted_dir):
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print("Extraction completed!")
    else:
        print("Dataset already extracted.")
    
    return os.path.join(extracted_dir, RAW_DATA_FILE)


def load_raw_data(filepath):
    """Load raw WISDM data from text file."""
    print(f"Loading raw data from {filepath}...")
    
    # Read the data file
    column_names = ['user_id', 'activity', 'timestamp', 'x', 'y', 'z']
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Clean the line (remove semicolons and fix formatting)
            line = line.strip().rstrip(';')
            if line:
                try:
                    parts = line.split(',')
                    if len(parts) == 6:
                        # Check if all fields are non-empty
                        if all(part.strip() for part in parts):
                            # Clean each part
                            cleaned_parts = [part.strip() for part in parts]
                            data.append(cleaned_parts)
                except:
                    continue
    
    df = pd.DataFrame(data, columns=column_names)
    
    # Convert data types with error handling
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Convert to proper types
    df['user_id'] = df['user_id'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    
    print(f"Loaded {len(df)} samples from {df['user_id'].nunique()} users")
    print(f"Activities: {df['activity'].unique()}")
    
    return df


def create_segments(df, window_size=200, step_size=100):
    """Create time series segments from raw data using sliding window."""
    print(f"Creating segments with window_size={window_size}, step_size={step_size}...")
    
    segments = []
    labels = []
    user_ids = []
    
    # Group by user and activity
    for (user, activity), group in df.groupby(['user_id', 'activity']):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Extract accelerometer data
        data = group[['x', 'y', 'z']].values
        
        # Create segments using sliding window
        for i in range(0, len(data) - window_size + 1, step_size):
            segment = data[i:i + window_size]
            if len(segment) == window_size:
                segments.append(segment)
                labels.append(activity)
                user_ids.append(user)
    
    print(f"Created {len(segments)} segments")
    return np.array(segments), np.array(labels), np.array(user_ids)


def preprocess_data(raw_data_path, window_size=200, step_size=100):
    """Preprocess WISDM dataset."""
    # Load raw data
    df = load_raw_data(raw_data_path)
    
    # Create segments
    X, y, users = create_segments(df, window_size, step_size)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y_encoded.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Save preprocessed data
    preprocessed_path = os.path.join(DATA_DIR, "preprocessed_data.npz")
    np.savez(
        preprocessed_path,
        X=X,
        y=y_encoded,
        users=users,
        classes=label_encoder.classes_
    )
    
    # Save label encoder
    encoder_path = os.path.join(DATA_DIR, "label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Preprocessed data saved to {preprocessed_path}")
    
    return X, y_encoded, users, label_encoder


def partition_data(X, y, users, num_partitions=10, split_by_user=True):
    """Partition data for federated learning."""
    print(f"Partitioning data into {num_partitions} partitions...")
    
    partitions = []
    
    if split_by_user:
        # Partition by user (more realistic for federated learning)
        unique_users = np.unique(users)
        users_per_partition = len(unique_users) // num_partitions
        
        for i in range(num_partitions):
            start_user = i * users_per_partition
            end_user = (i + 1) * users_per_partition if i < num_partitions - 1 else len(unique_users)
            partition_users = unique_users[start_user:end_user]
            
            # Get data for these users
            mask = np.isin(users, partition_users)
            X_partition = X[mask]
            y_partition = y[mask]
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X_partition, y_partition, test_size=0.2, random_state=42, stratify=y_partition
            )
            
            partitions.append({
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'users': partition_users
            })
            
            print(f"Partition {i}: {len(X_train)} train samples, {len(X_test)} test samples")
    else:
        # Random IID partition
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        partition_size = len(indices) // num_partitions
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < num_partitions - 1 else len(indices)
            partition_indices = indices[start_idx:end_idx]
            
            X_partition = X[partition_indices]
            y_partition = y[partition_indices]
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X_partition, y_partition, test_size=0.2, random_state=42, stratify=y_partition
            )
            
            partitions.append({
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            })
            
            print(f"Partition {i}: {len(X_train)} train samples, {len(X_test)} test samples")
    
    # Save partitions
    partitions_dir = os.path.join(DATA_DIR, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)
    
    for i, partition in enumerate(partitions):
        partition_path = os.path.join(partitions_dir, f"partition_{i}.npz")
        np.savez(partition_path, **partition)
    
    print(f"Partitions saved to {partitions_dir}")
    
    return partitions


def setup_wisdm_dataset(num_partitions=10, window_size=200, step_size=100, split_by_user=True):
    """Complete setup: download, preprocess, and partition WISDM dataset."""
    print("=" * 60)
    print("WISDM Dataset Setup")
    print("=" * 60)
    
    # Step 1: Download
    raw_data_path = download_wisdm()
    
    # Step 2: Preprocess
    X, y, users, label_encoder = preprocess_data(raw_data_path, window_size, step_size)
    
    # Step 3: Partition
    partitions = partition_data(X, y, users, num_partitions, split_by_user)
    
    print("=" * 60)
    print("Setup completed!")
    print("=" * 60)
    
    return partitions


if __name__ == "__main__":
    # Run the complete setup
    setup_wisdm_dataset(num_partitions=10, window_size=200, step_size=100, split_by_user=True)
