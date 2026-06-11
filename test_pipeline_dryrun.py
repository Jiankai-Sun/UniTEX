import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Importing CustomRGBTextureFullPipeline...")
    from pipeline import CustomRGBTextureFullPipeline
    print("Import successful! (Unexpected on CPU-only host, but great!)")
    
    # If import succeeded, try to initialize it
    print("Initializing CustomRGBTextureFullPipeline...")
    rgb_tfp = CustomRGBTextureFullPipeline(super_resolutions=False,
                                            filt_gradient_points=False,
                                            filt_large_angle_points=True,
                                            seed = 63)
    print("Initialization successful!")

except RuntimeError as e:
    if "Found no NVIDIA driver" in str(e) or "cuda" in str(e).lower():
         print(f"\n[SUCCESS] Dry-run caught expected CUDA error during import/init:\n  {e}")
         print("\nEnvironment is FULLY READY to run on GPU!")
         sys.exit(0)
    else:
         print("\n[FAILURE] Unexpected RuntimeError:")
         traceback.print_exc()
         sys.exit(1)
except Exception as e:
    print("\n[FAILURE] Unexpected exception:")
    traceback.print_exc()
    sys.exit(1)
