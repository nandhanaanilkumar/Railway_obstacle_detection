import os
import sys

# Ensure current working directory and Python path include the 'files' directory
files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
if files_dir not in sys.path:
    sys.path.insert(0, files_dir)
os.chdir(files_dir)

# Run the main app.py inside files directory
app_path = os.path.join(files_dir, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, app_path, "exec"))
