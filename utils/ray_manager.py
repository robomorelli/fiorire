"""
Ray cluster management utilities.
Handles initialization, cleanup, and graceful shutdown.
"""

import ray
import atexit
import signal
import sys
import time
import ray


_RAY_INITIALIZED = False
_CLEANUP_REGISTERED = False
_INTERRUPTED = False  # ← NUOVO: flag per interruzione


def initialize_ray_cluster(address='auto', ignore_reinit_error=True):
    """
    Initialize Ray cluster with automatic cleanup on exit.

    Args:
        address: Ray cluster address ('auto' for existing cluster, None for local)
        ignore_reinit_error: Whether to ignore re-initialization errors

    Returns:
        dict with cluster info
    """
    global _RAY_INITIALIZED, _CLEANUP_REGISTERED

    if _RAY_INITIALIZED:
        print("⚠️  Ray already initialized")
        return ray.cluster_resources()

    print("\n" + "=" * 80)
    print("🚀 INITIALIZING RAY CLUSTER")
    print("=" * 80)

    try:
        ray.init(address=address, ignore_reinit_error=ignore_reinit_error)
        _RAY_INITIALIZED = True

        # Register cleanup handlers
        if not _CLEANUP_REGISTERED:
            _register_cleanup_handlers()
            _CLEANUP_REGISTERED = True

        # Print cluster info
        cluster_resources = ray.cluster_resources()
        print(f"✅ Ray initialized successfully")
        print(f"   - CPUs: {cluster_resources.get('CPU', 0)}")
        print(f"   - GPUs: {cluster_resources.get('GPU', 0)}")
        print(f"   - Memory: {cluster_resources.get('memory', 0) / 1024**3:.1f} GB")
        print("=" * 80 + "\n")

        return cluster_resources

    except Exception as e:
        print(f"❌ Failed to initialize Ray: {e}")
        raise


def shutdown_ray_cluster(verbose=True):
    """
    Shutdown Ray cluster gracefully.

    Args:
        verbose: Whether to print shutdown messages
    """
    global _RAY_INITIALIZED

    if not _RAY_INITIALIZED:
        if verbose:
            print("ℹ️  Ray not initialized, nothing to shutdown")
        return

    if verbose:
        print("\n" + "=" * 80)
        print("🛑 SHUTTING DOWN RAY CLUSTER")
        print("=" * 80)

    try:
        ray.shutdown()
        _RAY_INITIALIZED = False

        if verbose:
            print("✅ Ray shutdown successfully")
            print("=" * 80 + "\n")

    except Exception as e:
        if verbose:
            print(f"⚠️  Error during Ray shutdown: {e}")


def _register_cleanup_handlers():
    """Register cleanup handlers for graceful shutdown."""

    # Register atexit handler
    atexit.register(_cleanup_on_exit)

    # Register signal handlers
    signal.signal(signal.SIGINT, _cleanup_on_signal)   # Ctrl+C
    signal.signal(signal.SIGTERM, _cleanup_on_signal)  # Kill

    # Unix-specific signals
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, _cleanup_on_signal)


def _cleanup_on_exit():
    """Cleanup handler called on normal exit."""
    shutdown_ray_cluster(verbose=False)


def _cleanup_on_signal(signum, frame):
    """Cleanup handler called on signal interrupt."""
    global _INTERRUPTED

    if _INTERRUPTED:
        # Already handling interrupt, don't re-enter
        return

    _INTERRUPTED = True

    signal_name = signal.Signals(signum).name

    print(f"\n\n⚠️  Received signal {signal_name}, cleaning up...")
    shutdown_ray_cluster(verbose=True)

    print(f"👋 Goodbye!")

    # ✅ FIX: Non chiamare sys.exit() - lascia che Python termini naturalmente
    # Questo è compatibile con debugger come PyCharm
    raise KeyboardInterrupt  # ← Invece di sys.exit(0)


def get_cluster_info():
    """
    Get current Ray cluster information.

    Returns:
        dict with cluster resources or None if not initialized
    """
    if not _RAY_INITIALIZED:
        return None

    try:
        return {
            'resources': ray.cluster_resources(),
            'available': ray.available_resources(),
            'nodes': len(ray.nodes()),
        }
    except Exception:
        return None


def is_ray_initialized():
    """Check if Ray is initialized."""
    return _RAY_INITIALIZED


"""
Ray environment setup and configuration.
"""

import ray
import shutil
import json
from pathlib import Path


# utils/ray_setup.py

def setup_and_initialize_ray(address='auto', password=None, object_store_memory_gb=30):
    """
    Setup Ray environment with proper cleanup and configuration.

    Args:
        address: Ray cluster address
            - 'auto': Connect to existing cluster or start local
            - None: Start local cluster only
            - 'ray://IP:PORT' or 'IP:PORT': Connect to specific cluster
        password: Redis password for cluster authentication
        object_store_memory_gb: Object store memory in GB (only for local cluster)
    """
    print("\n" + "=" * 80)
    print("🔧 SETTING UP RAY ENVIRONMENT")
    print("=" * 80)

    # 1. Clean /tmp/ray (only if local)
    is_local = (address is None)

    if is_local:
        print("\n1️⃣ Cleaning Ray temp directory...")
        ray_tmp = Path("/tmp/ray")
        if ray_tmp.exists():
            try:
                shutil.rmtree(ray_tmp)
                print(f"   ✓ Cleaned: {ray_tmp}")
            except Exception as e:
                print(f"   ⚠️  Could not clean {ray_tmp}: {e}")
    else:
        print("\n1️⃣ Connecting to existing Ray cluster...")
        print(f"   - Address: {address}")
        if password:
            print(f"   - Password: {'*' * len(password)}")

    # 2. Setup alternative directories (only for local)
    if is_local:
        ray_temp_dir = Path.home() / "ray_temp"
        ray_temp_dir.mkdir(exist_ok=True)

        ray_spill_dir = Path.home() / "ray_spill"
        ray_spill_dir.mkdir(exist_ok=True)

        print(f"\n2️⃣ Using alternative directories:")
        print(f"   - Temp: {ray_temp_dir}")
        print(f"   - Spill: {ray_spill_dir}")

    # 3. Initialize Ray
    print(f"\n{'3️⃣' if is_local else '2️⃣'} Initializing Ray...")

    # ✅ Build init kwargs
    init_kwargs = {
        "ignore_reinit_error": True,
    }

    # Add address if provided
    if address:
        init_kwargs["address"] = address

    # ✅ Add password if provided
    if password:
        init_kwargs["_redis_password"] = password

    # ✅ Only for LOCAL cluster
    if is_local:
        init_kwargs["_temp_dir"] = str(ray_temp_dir)
        init_kwargs["object_store_memory"] = object_store_memory_gb * 1024 ** 3
        init_kwargs["_system_config"] = {
            "automatic_object_spilling_enabled": True,
            "max_io_workers": 4,
            "object_spilling_config": json.dumps({
                "type": "filesystem",
                "params": {
                    "directory_path": str(ray_spill_dir)
                }
            })
        }
        print(f"   ℹ️  Starting LOCAL Ray cluster")
    else:
        print(f"   ℹ️  Connecting to PBS Ray cluster")

    # Initialize
    ray.init(**init_kwargs)

    print(f"   ✓ Ray initialized")
    if is_local:
        print(f"   ✓ Object store: {object_store_memory_gb}GB")

    # Print cluster info
    print(f"\n   📊 Cluster info:")
    print(f"      - Nodes: {len(ray.nodes())}")
    print(f"      - CPUs: {ray.cluster_resources().get('CPU', 0)}")
    print(f"      - GPUs: {ray.cluster_resources().get('GPU', 0)}")

    print("=" * 80 + "\n")

def shutdown_ray():
    """
    Gracefully shutdown Ray.
    """
    print("\n🧹 Shutting down Ray...")
    try:
        ray.shutdown()
        print("   ✓ Ray shutdown complete")
    except Exception as e:
        print(f"   ⚠️  Error during shutdown: {e}")