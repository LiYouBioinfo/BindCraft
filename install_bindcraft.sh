#!/bin/bash
################## BindCraft installation script (critical-minimal)

# Default args
pkg_manager='conda'
cuda=''

# Prefer mamba for heavy ops if available
if [ "$pkg_manager" = "conda" ] && command -v mamba >/dev/null 2>&1; then
  pkg_manager="mamba"
fi

# Parse CLI options
OPTIONS=p:c:
LONGOPTIONS=pkg_manager:,cuda:
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"
while true; do
  case "$1" in
    -p|--pkg_manager) pkg_manager="$2"; shift 2 ;;
    -c|--cuda)        cuda="$2";        shift 2 ;;
    --) shift; break ;;
    *) echo -e "Invalid option $1" >&2; exit 1 ;;
  esac
done

echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"

############################################################################################################
SECONDS=0
install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

### Create env
echo -e "Installing BindCraft environment\n"
$pkg_manager create --name BindCraft python=3.10 -y || { echo -e "Error: Failed to create BindCraft conda environment"; exit 1; }
conda env list | grep -w 'BindCraft' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'BindCraft' does not exist after creation."; exit 1; }

### Activate env
echo -e "Loading BindCraft environment\n"
source "${CONDA_BASE}/bin/activate" "${CONDA_BASE}/envs/BindCraft" || { echo -e "Error: Failed to activate the BindCraft environment."; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "BindCraft" ] || { echo -e "Error: The BindCraft environment is not active."; exit 1; }
echo -e "BindCraft environment activated at ${CONDA_BASE}/envs/BindCraft"

# ---- CRITICAL: fast solver + strict priority + stable CUDA override ----
conda install -n base -c conda-forge -y conda-libmamba-solver >/dev/null 2>&1 || true
conda config --set solver libmamba
conda config --set channel_priority strict
if [ -n "$cuda" ]; then export CONDA_OVERRIDE_CUDA="$cuda"; else export CONDA_OVERRIDE_CUDA="12.2"; fi
# -----------------------------------------------------------------------

### Core packages (no graylab channel here)
echo -e "Instaling conda requirements (core)\n"
$pkg_manager install -y \
  -c conda-forge -c nvidia \
  pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg fsspec py3dmol \
  chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
  'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0' cuda-nvcc cudnn \
|| { echo -e "Error: Failed to install core conda packages."; exit 1; }

### PyRosetta separately (graylab), avoids solver stalls
echo -e "Installing PyRosetta (separate step)\n"
$pkg_manager install -y -c https://conda.graylab.jhu.edu pyrosetta \
|| { echo -e "Error: Failed to install PyRosetta"; exit 1; }

### Verify required packages
required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
missing_packages=()
for pkg in "${required_packages[@]}"; do
  conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done
if [ ${#missing_packages[@]} -ne 0 ]; then
  echo -e "Error: The following packages are missing from the environment:"
  for pkg in "${missing_packages[@]}"; do echo -e " - $pkg"; done
  exit 1
fi

### ColabDesign (use env's pip)
echo -e "Installing ColabDesign\n"
python -m pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps \
  || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }

### AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

### Permissions
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

### Finish
conda deactivate
echo -e "BindCraft environment set up\n"

### Cleanup caches
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

t=$SECONDS
echo -e "Successfully finished BindCraft installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate BindCraft\""
echo -e "\nInstallation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
