# Git LFS Setup

This repository contains large files that should be tracked with Git LFS (Large File Storage).

## Installation

Before cloning or pulling this repository, install Git LFS:

### Linux
```bash
sudo apt-get install git-lfs
git lfs install
```

### macOS
```bash
brew install git-lfs
git lfs install
```

### Windows
Download and install from https://git-lfs.github.com/
Then run:
```bash
git lfs install
```

## Files tracked with LFS

The following file types are tracked with Git LFS:
- Model files (*.pt, *.pth, *.pkl, *.h5)
- Large datasets (*.npy, *.npz)
- Audio files (*.wav, *.mp3)

## Checking if Git LFS is working

After cloning the repository, you can verify Git LFS is working by running:
```bash
git lfs ls-files
```

## If you've already cloned without Git LFS

If you already cloned the repository without Git LFS, run:
```bash
git lfs install
git lfs pull
```

This will download all LFS-tracked files properly. 