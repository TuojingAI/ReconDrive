

# Getting Started

## 1. Create Conda Environment

```bash
conda create -n recondrive python=3.10
conda activate recondrive
```

---

## 2. Install Dependencies

### Step 1: Automatic Installation

```bash
python -m pip install -r requirements.txt
```

---

### Step 2: Manual Installation

#### (1) Install PyTorch3D

```bash
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2
conda install ./pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2
```

---

#### (2) Install Gaussian Splatting Dependencies

```bash
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

---

#### (3) Install SAM2

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Download SAM2 checkpoint
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```


