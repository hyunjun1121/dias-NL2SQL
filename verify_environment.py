"""
Verify conda environment setup for Qwen models.
Run this after environment setup to check all dependencies.
"""

import sys
import importlib
from pathlib import Path

def check_package(package_name, min_version=None, import_name=None):
    """Check if a package is installed and meets version requirements."""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            from packaging import version as v
            if v.parse(version) >= v.parse(min_version):
                print(f"✓ {package_name}: {version} (>= {min_version})")
                return True
            else:
                print(f"⚠ {package_name}: {version} (need >= {min_version})")
                return False
        else:
            print(f"✓ {package_name}: {version}")
            return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"? {package_name}: Error - {e}")
        return False

def check_cuda():
    """Check CUDA availability and version."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: Available")
            print(f"  - Version: {torch.version.cuda}")
            print(f"  - GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ CUDA: Not available")
            return False
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False

def check_local_modules():
    """Check if local project modules are importable."""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    modules_to_check = [
        ('utils.llm_client', 'LLMClient'),
        ('utils.database_connector', 'DatabaseConnector'),
        ('model.subtask_extractor', 'ConfidentSubTaskExtractor'),
        ('config.config', 'Config'),
    ]

    all_ok = True
    for module_path, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                print(f"✓ {module_path}.{class_name}: OK")
            else:
                print(f"⚠ {module_path}: Missing {class_name}")
                all_ok = False
        except ImportError as e:
            print(f"✗ {module_path}: {e}")
            all_ok = False

    return all_ok

def main():
    print("="*60)
    print("Qwen NL2SQL Environment Verification")
    print("="*60)
    print()

    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()

    # Core packages
    print("Core Packages:")
    print("-"*40)
    core_ok = all([
        check_package('torch', '2.0.0'),
        check_package('transformers', '4.51.0'),
        check_package('accelerate', '0.25.0'),
    ])
    print()

    # CUDA check
    print("GPU/CUDA Status:")
    print("-"*40)
    cuda_ok = check_cuda()
    print()

    # Model-specific packages
    print("Model-Specific Packages:")
    print("-"*40)
    model_ok = all([
        check_package('sentencepiece'),
        check_package('protobuf'),
        check_package('einops'),
        check_package('safetensors'),
        check_package('tokenizers'),
    ])
    print()

    # NL2SQL packages
    print("NL2SQL Pipeline Packages:")
    print("-"*40)
    nl2sql_ok = all([
        check_package('sqlalchemy'),
        check_package('pandas'),
        check_package('numpy'),
        check_package('langchain'),
        check_package('langchain_community'),
        check_package('langchain_core'),
        check_package('ollama'),
    ])
    print()

    # Utility packages
    print("Utility Packages:")
    print("-"*40)
    util_ok = all([
        check_package('datasketch'),
        check_package('chromadb'),
        check_package('func_timeout', import_name='func_timeout'),
        check_package('psutil'),
        check_package('tqdm'),
    ])
    print()

    # Optional optimization packages
    print("Optional Optimization Packages:")
    print("-"*40)
    check_package('optimum')
    check_package('bitsandbytes')
    check_package('flash_attn')
    check_package('xformers')
    print()

    # Local modules
    print("Local Project Modules:")
    print("-"*40)
    local_ok = check_local_modules()
    print()

    # Summary
    print("="*60)
    print("Summary:")
    print("-"*40)

    status = {
        "Core packages": core_ok,
        "CUDA": cuda_ok,
        "Model packages": model_ok,
        "NL2SQL packages": nl2sql_ok,
        "Utility packages": util_ok,
        "Local modules": local_ok,
    }

    for component, ok in status.items():
        status_str = "✓ OK" if ok else "✗ FAILED"
        print(f"{component}: {status_str}")

    all_critical_ok = all([core_ok, model_ok, nl2sql_ok, local_ok])

    print()
    if all_critical_ok:
        print("✓ Environment is ready for Qwen model testing!")
        if not cuda_ok:
            print("  (Note: CUDA not available, will run on CPU)")
    else:
        print("✗ Some critical packages are missing.")
        print("  Run: bash setup_conda_fresh.sh")

    print("="*60)

    return 0 if all_critical_ok else 1

if __name__ == "__main__":
    sys.exit(main())