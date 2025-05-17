import os

def save_labels_to_text_file(directory, output_file):
    # Get all folder names in the given directory
    labels = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

    # Write the labels to a text file
    with open(output_file, 'w') as file:
        for i, label in enumerate(labels):
            if i == len(labels) - 1:
                file.write(label)  # Don't add a newline after the last label
            else:
                file.write(label + '\n')  # Add a newline after each label except the last one

    print(f"Labels saved to {output_file}")

# Example usage
directory_path = './data/'  # Replace with your directory path
output_file = './model/labels.txt'  # Replace with your desired output file name

save_labels_to_text_file(directory_path, output_file)