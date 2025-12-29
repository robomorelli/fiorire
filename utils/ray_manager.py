"""
Ray cluster management utilities.
Handles initialization, cleanup, and graceful shutdown.
"""

import ray
import atexit
import signal
import sys
import time


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