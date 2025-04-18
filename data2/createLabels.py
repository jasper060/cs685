import os
import csv
import glob

def create_image_label_csv(root_folder):
    """
    Creates a CSV file where each row contains the image path and its corresponding class label.
    The class label is determined by the name of the subfolder containing the image.

    Args:
        root_folder (str): The path to the root folder containing the subfolders (e.g., 'calling', 'clapping', etc.).
        csv_filename (str, optional): The name of the CSV file to create. Defaults to "image_labels.csv".
    """

    class_names = [
        "calling",
        "clapping",
        "cycling",
        "dancing",
        "drinking",
        "eating",
        "fighting",
        "hugging",
        "laughing",
        "listening_to_music",
        "running",
        "sitting",
        "sleeping",
        "texting",
        "using_laptop",
    ]
    # Create a dictionary to map class names to integer labels
    class_to_label = {class_name: label for label, class_name in enumerate(class_names)}

    csv_filename = f"{root_folder}_labels.csv"

    print(csv_filename)

    # Construct the full path for the CSV file, placing it at the same level as the subfolders.
    csv_path = os.path.join(root_folder, csv_filename)

    # Open the CSV file in write mode
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(["filename", "label", "label_idx", "image_num"])

        # Iterate through each subfolder (class)
        for class_name, label in class_to_label.items():
            class_folder_path = os.path.join(root_folder, class_name)
            # Use glob to get all image files.  This handles different image extensions.
            image_files = glob.glob(os.path.join(class_folder_path, "*"))

            # Iterate through each image in the subfolder
            for idx, image_path in enumerate(image_files):
                filename = image_path.split(os.sep)[-1]
                label = image_path.split(os.sep)[-2]
                label_idx = class_to_label[label]
                image_num = idx
                # Write the image path and its corresponding label to the CSV file
                writer.writerow([filename, label, label_idx, image_num])

    print(f"CSV file '{csv_path}' has been created successfully.")


if __name__ == "__main__":
    root_folder = "data2"
    for dir_name in os.listdir():
        if (dir_name.endswith(".py")):
            continue
        create_image_label_csv(dir_name)
