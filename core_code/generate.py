import os 

snapshot_directory = "Run_0/"
current_dir = os.getcwd()
gparent_dir=os.path.dirname(current_dir)

relevant_path=os.path.join(gparent_dir,snapshot_directory)

def replace_and_save(input_file_path, output_file_path, old_words, new_words):
    try:
        # Read the content of the original Bash script
        with open(input_file_path, 'r') as f:
            script_content = f.read()
        
        # Replace the old word with the new word
       # Replace old words with corresponding new words
        modified_content = script_content.replace(old_words[1], new_words[1]).replace(old_words[0], new_words[0])
       
        # Construct the output file path
        output_file_path = os.path.join(output_file_path, f"modified_{os.path.basename(input_file_path)}")


        # Write the modified content to the new file
        with open(output_file_path, 'w') as f:
            f.write(modified_content)

        print(f"Script saved as: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


def generate_juv_strings():
    snapshot_strings = []

    for i in range(3):
        # Use f-strings to format the strings
        snapshot_string = f"planet0_{i:05d}"
        snapshot_strings.append(snapshot_string)

    return snapshot_strings

# Example usage
generated_strings = generate_juv_strings()

def remove_leading_zeros(input_string):
    # Split the string into two parts based on the underscore
    parts = input_string.split('_')

    # If there are two parts and the second part is numeric
    if len(parts) == 2 and parts[1].isdigit():
        # Remove leading zeros from the second part
        second_part = str(int(parts[1]))

        # Join the parts back together
        result = f"{parts[0]}_{second_part}"

        return result
    else:
        # If the format is not as expected, return the original string
        return input_string
    
for snapshot in generated_strings:
    #check if the snapshot exists, if yes create a result directory
    if os.path.exists(relevant_path+snapshot):
        #if snapshot does exist, create a new directory for its job script and results from mcfost
        
          new_directory = f"./{remove_leading_zeros(snapshot)}"
          if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            #once the new directory is created create a job script and save
            o_words = ["replace_me", "path_name"]
            n_words = [relevant_path+snapshot, remove_leading_zeros(snapshot)]
            replace_and_save('./submit.sh',new_directory,o_words,n_words)
            print(f"New directory {new_directory} created.")
          else:
            print(f"The directory {new_directory} already exists.")
    else:
        print(f"could not find file in path {relevant_path+snapshot}")
