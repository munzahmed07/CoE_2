import sys
import os

print("--- DIAGNOSTIC INFORMATION ---")

# 1. Print the Current Working Directory
current_working_directory = os.getcwd()
print(f"\n[1] Current Working Directory:\n{current_working_directory}")

# 2. Print the contents of this directory
print("\n[2] Contents of this Directory:")
try:
    contents = os.listdir(current_working_directory)
    if not contents:
        print("    (This directory is empty)")
    else:
        for item in contents:
            print(f"    - {item}")
except Exception as e:
    print(f"    Could not list directory contents: {e}")

# 3. Print Python's System Path (where it looks for modules)
print("\n[3] Python's System Path (sys.path):")
for path_item in sys.path:
    print(f"    - {path_item}")

print("\n--- END OF DIAGNOSTICS ---")
