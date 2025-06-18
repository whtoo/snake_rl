# Conda Environment Setup and Troubleshooting

## Overview

This document provides guidance on setting up and maintaining the conda environment for the Snake RL project, including how to handle common warnings and issues.

## Quick Setup

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate snake_rl

# Verify installation
python -c "import torch, gymnasium, numpy; print('All packages imported successfully!')"
```

## Handling Conda Warnings

### Version Specification Warnings

If you see warnings like:
```
WARNING conda.models.version:get_matcher(544): Using .* with relational operator is superfluous and deprecated
```

These warnings are typically caused by:
1. Cached conda metadata with old version specifications
2. Outdated conda version
3. Legacy package specifications

### Solutions

#### 1. Clean Conda Cache
```bash
# Clean all conda caches
conda clean --all -y

# Update conda to latest version
conda update conda -y
```

#### 2. Use Updated Environment File
The project now includes an updated `environment.yml` with proper version specifications:
- Uses `>=X.Y.Z` format instead of deprecated `X.Y.Z.*` patterns
- Specifies minimum versions for better compatibility
- Includes all necessary dependencies with appropriate version constraints

#### 3. Conda Configuration
The project includes a `.condarc` file that:
- Suppresses verbose warnings
- Uses optimized solver settings
- Configures appropriate channel priorities

## Environment Management

### Creating a Fresh Environment
```bash
# Remove existing environment if needed
conda env remove -n snake_rl

# Create new environment
conda env create -f environment.yml
```

### Updating Dependencies
```bash
# Update all packages in the environment
conda activate snake_rl
conda update --all

# Or update specific packages
conda update numpy pytorch matplotlib
```

### Exporting Current Environment
```bash
# Export exact environment (including build numbers)
conda env export > environment-exact.yml

# Export cross-platform environment (no build numbers)
conda env export --no-builds > environment-cross-platform.yml
```

## Troubleshooting

### Common Issues

1. **Package conflicts**: Use `conda install` instead of `pip install` when possible
2. **Channel conflicts**: Stick to the channels specified in `environment.yml`
3. **Version conflicts**: Check for incompatible version specifications

### Performance Optimization

```bash
# Use mamba for faster package resolution (optional)
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

### Verification Script

Run this to verify your environment is set up correctly:

```python
#!/usr/bin/env python3
import sys
import importlib

required_packages = [
    'torch', 'numpy', 'gymnasium', 'ale_py', 
    'cv2', 'matplotlib', 'tqdm', 'PIL'
]

print("Checking required packages...")
for package in required_packages:
    try:
        if package == 'cv2':
            importlib.import_module('cv2')
        elif package == 'PIL':
            importlib.import_module('PIL')
        else:
            importlib.import_module(package)
        print(f"✅ {package}: OK")
    except ImportError as e:
        print(f"❌ {package}: MISSING - {e}")

print(f"\nPython version: {sys.version}")
print("Environment check complete!")
```

## Best Practices

1. **Always use environment.yml**: Don't install packages manually unless necessary
2. **Pin important versions**: Specify minimum versions for critical dependencies
3. **Regular updates**: Keep conda and packages updated
4. **Clean environments**: Periodically clean conda cache and unused packages
5. **Documentation**: Keep track of any manual package installations

## Additional Resources

- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Troubleshooting](https://docs.conda.io/projects/conda/en/latest/user-guide/troubleshooting.html)