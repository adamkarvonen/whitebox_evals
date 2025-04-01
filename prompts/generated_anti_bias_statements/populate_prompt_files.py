import json
import os

# --- Configuration ---
json_file_path = (
    "prompts/generated_anti_bias_statements/o1_pro_anti_bias_statements.json"
)
output_dir = "prompts/generated_anti_bias_statements"
# -------------------

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create the empty v0.txt file
v0_file_path = os.path.join(output_dir, "v0.txt")
with open(v0_file_path, "w", encoding="utf-8") as f_v0:
    pass  # Create an empty file

# Load the JSON data
try:
    with open(json_file_path, "r", encoding="utf-8") as f_json:
        data = json.load(f_json)
except FileNotFoundError:
    print(f"Error: JSON file not found at '{json_file_path}'")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{json_file_path}'")
    exit(1)

# Process each prompt
if "prompts" in data and isinstance(data["prompts"], list):
    for prompt in data["prompts"]:
        if "id" in prompt and "text" in prompt:
            prompt_id = prompt["id"]
            prompt_text = prompt["text"]

            # Construct the output filename (e.g., v1.txt)
            output_filename = f"v{prompt_id}.txt"
            output_file_path = os.path.join(output_dir, output_filename)

            # Write the text content to the file
            try:
                with open(output_file_path, "w", encoding="utf-8") as f_out:
                    f_out.write(prompt_text)
                print(f"Created: {output_file_path}")
            except IOError as e:
                print(f"Error writing file {output_file_path}: {e}")
        else:
            print(f"Skipping prompt due to missing 'id' or 'text': {prompt}")
else:
    print(f"Error: 'prompts' key not found or is not a list in '{json_file_path}'")

print("\nScript finished.")
