import json
import subprocess
import shutil
import os

def run_dependency_analysis(config_file, go_file):
    current_directory = os.getcwd()
    all_outputs = []  # List to hold all the outputs

    # Load the JSON configuration
    with open(config_file, 'r') as file:
        data = json.load(file)
    
    # Iterate over each repo entry in the JSON configuration
    for entry in data['go']:
        repo = entry['repo']
        commit_sha = entry['commit_sha']
        entrypoint_path = entry['entrypoint_path']
        topic = entry['topic']
        
        # Define the directory to clone into and where to copy the Go file
        clone_dir = f"./clones/{repo.replace('/', '_')}"
        entrypoint_dir = f"{clone_dir}/{entrypoint_path}"
        
        # Clone the repository at the specific commit
        subprocess.run(['git', 'clone', f'https://github.com/{repo}.git', clone_dir])
        subprocess.run(['git', '-C', clone_dir, 'checkout', commit_sha])
        
        # Ensure the entrypoint directory exists
        os.makedirs(entrypoint_dir, exist_ok=True)
        
        # Copy the Go dependency analysis file to the entrypoint directory
        shutil.copy(go_file, clone_dir)
        
        # Change to the entrypoint directory
        os.chdir(clone_dir)
        
        # Build and run the Go program
        subprocess.run(['go', 'build', '-o', 'dependency_analysis', 'dependency_analysis.go'])
        print(f"Running dependency analysis for {topic} in {repo}")
        subprocess.run(['./dependency_analysis', entrypoint_path, repo.split('/')[-1]])
        
        # Read the output.json and parse it
        with open('output.json', 'r') as output_file:
            output_data = json.load(output_file)
            all_outputs.append({output_data['name']: output_data['repoName']})
            
        
        # Change back to the original directory
        os.chdir(current_directory)
    shutil.rmtree("./clones/")
    # Write all output data to a file
    all_outputs = {"Go": all_outputs}
    with open('./data/go_dependecy.json', 'w') as f:
        json.dump(all_outputs, f, indent=4)

if __name__ == "__main__":
    config_file = '../../cherrypick/lists.json'  # Update with the actual path to your JSON config file
    go_file = '../dependency_analysis.go'  # Path to your Go file
    run_dependency_analysis(config_file, go_file)
