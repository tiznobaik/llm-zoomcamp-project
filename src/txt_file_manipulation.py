import os
import re
import shutil

# Function to reformat the content of the file
def reformat_file_content(content):
    # Remove any blank lines
    content = '\n'.join([line for line in content.splitlines() if line.strip()])

    # Add separating lines ("---") between section headings (e.g., "4.1", "4.2", etc.)
    content = re.sub(r'(\d+\.\d+)', r'---\n\1', content)
    
    # Ensure introduction has a separating line too
    content = re.sub(r'(Introduction)', r'---\n\1\n---', content)
    
    # Ensure conclusion has a separating line at the end
    content = re.sub(r'(Conclusion)', r'---\n\1\n---', content)
    
    return content

# Function to load the content of a file
def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Function to save the formatted content back to a file
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

# Function to process all txt files in a source directory and save to a destination directory
def process_files_in_directory(source_directory, destination_directory):
    # Ensure destination directory exists, if not, create it
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    files_data = []

    # Get all .txt files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith(".txt"):
            # Extract the chapter number from the filename (assumes the filename has 'Chapter X' format)
            chapter_num = int(re.search(r'\d+', filename).group())

            input_filepath = os.path.join(source_directory, filename)
            output_filepath = os.path.join(destination_directory, 'formatted_' + filename)

            # Load the file content
            file_content = load_file(input_filepath)

            # Reformat the file content
            formatted_content = reformat_file_content(file_content)

            # Save the formatted content back to the destination directory
            save_file(output_filepath, formatted_content)

            # Store the formatted content along with its chapter number
            files_data.append((chapter_num, formatted_content))

            print(f"Formatted {filename} and saved as formatted_{filename}")

    # Sort the files based on the chapter number (ascending)
    files_data.sort(key=lambda x: x[0])

    # Combine the contents of all formatted files into one
    combined_content = "\n\n".join([content for _, content in files_data])

    # Save the combined content to a single file in the destination directory
    combined_filepath = os.path.join(destination_directory, 'fcra_doc.txt')
    save_file(combined_filepath, combined_content)

    print(f"Combined all files and saved as combined_formatted.txt")

# Set the directory paths
source_directory = '../data/FCRA Data/raw'  # Replace with the path to your folder containing the .txt files
destination_directory = '../data/FCRA Data/final'  # Replace with the path to your folder for saving formatted files

# Process all .txt files in the source directory and save the formatted and combined files to the destination directory
process_files_in_directory(source_directory, destination_directory)
