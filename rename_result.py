import os

# Directory path containing the files
directory = "./Training_output/QAOCS/LOL_3D"

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Split the filename into parts
        parts = filename.split("_")
        
        # Extract the relevant parts
        dataset_name = parts[1].replace("3D","")
        trace_number = parts[3]
        
        # Construct the new filename
        new_filename = f"{dataset_name}_trace_{trace_number}.csv"
        
        # Get the full file paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        
        print(f"Renamed: {filename} -> {new_filename}")