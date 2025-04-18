import os
import shutil
import random

def create_eval_split(base_path=".", train_folder="train", eval_folder="eval", split_ratio=0.2):
    train_path = os.path.join(base_path, train_folder)
    eval_path = os.path.join(base_path, eval_folder)
    
    # Ensure eval directory exists
    os.makedirs(eval_path, exist_ok=True)
    
    # Iterate over subfolders in train
    for subfolder in os.listdir(train_path):
        subfolder_train_path = os.path.join(train_path, subfolder)
        subfolder_eval_path = os.path.join(eval_path, subfolder)
        
        # Ensure corresponding subfolder exists in eval
        os.makedirs(subfolder_eval_path, exist_ok=True)
        
        if os.path.isdir(subfolder_train_path):
            images = [img for img in os.listdir(subfolder_train_path) if os.path.isfile(os.path.join(subfolder_train_path, img))]
            
            # Determine the number of images to move
            num_to_move = int(len(images) * split_ratio)
            images_to_move = random.sample(images, num_to_move)
            
            # Move selected images
            for img in images_to_move:
                src = os.path.join(subfolder_train_path, img)
                dst = os.path.join(subfolder_eval_path, img)
                shutil.move(src, dst)
                
            print(f"Moved {num_to_move} images from {subfolder_train_path} to {subfolder_eval_path}")

if __name__ == "__main__":
    create_eval_split()
