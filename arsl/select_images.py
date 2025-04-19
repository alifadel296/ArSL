import argparse
import os
import zipfile


def select_images(args):
    max_frames_num = 0

    # Process images in all directories
    for dir_path in args.image_dirs:
        
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Image directory '{dir_path}' does not exist")

        # Process frames in directory
        sign_folders = sorted(os.listdir(dir_path))
        for sign_folder in sign_folders:
            
            sign_path = os.path.join(dir_path, sign_folder)
            sign_sessions = sorted(os.listdir(sign_path))
            
            for session in sign_sessions:
                
                num_frames = 0
                session_path = os.path.join(sign_path, session)
                images = sorted(os.listdir(session_path))
                
                for idx, img in enumerate(images):
                    
                    if idx % 10 == 0:
                        num_frames += 1    # Retain every 10th frame to downsample while preserving temporal key points
                        continue
                    
                    os.remove(os.path.join(session_path, img))
                    
                max_frames_num = max(max_frames_num, num_frames)

    # Compress processed directories
    with zipfile.ZipFile(args.archive_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dir_path in args.image_dirs:
            
            parent_dir = os.path.dirname(dir_path)
            for root, dirs, files in os.walk(dir_path):
            
                for file in files:
                    
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, parent_dir)
                    zipf.write(file_path, arcname)

    print(f"Successfully compressed to {args.archive_name}")
    
    return max_frames_num



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Select images and compress processed dataset")
    
    parser.add_argument(
        "-id",
        "--image_dirs",
        nargs="*",
        type= str,
        required=True,
        help="Input image directories to process",
    )
    
    parser.add_argument(
        "-an",
        "--archive_name",
        type= str,
        required=True,
        help="Output ZIP archive name/path",
    )

    args = parser.parse_args()

    select_images(args)
