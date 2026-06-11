import sys
import os
import argparse
from pipeline import CustomRGBTextureFullPipeline

def main():
    parser = argparse.ArgumentParser(description="Run UniTEX Texture Painting")
    parser.add_argument("--image_path", type=str, default="test_cases/coca_cola/ref_image.jpg", help="Path to reference image")
    parser.add_argument("--mesh_path", type=str, default="test_cases/coca_cola/inputmesh.obj", help="Path to input OBJ mesh")
    parser.add_argument("--output_dir", type=str, default="outputs/coca_cola", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Redirect stdout and stderr to GCS log file
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "stdout.log")
    log_file = open(log_path, "w", buffering=1) # Line buffering
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"Starting UniTEX pipeline...")
    print(f"  Image: {args.image_path}")
    print(f"  Mesh: {args.mesh_path}")
    print(f"  Output Dir: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    rgb_tfp = CustomRGBTextureFullPipeline(
        super_resolutions=False,
        filt_gradient_points=False,
        filt_large_angle_points=True,
        seed=63
    )
    
    rgb_tfp(args.output_dir, args.image_path, args.mesh_path, clear_cache=False)
    print("UniTEX pipeline finished successfully!")

if __name__ == "__main__":
    main()
