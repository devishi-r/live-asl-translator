import subprocess

def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{script_name} ran successfully.")
        print(result.stdout)
    else:
        print(f"Error running {script_name}.")
        print(result.stderr)

scripts = ['convert_videos_to_points.py', 'embed_database_words.py', 'debugging_test.py']

for script in scripts:
    run_script(script)
