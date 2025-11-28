import json
import os

notebook_path = "mimic-iv-ext-direct-1.0.0/convertor.ipynb"

new_code = r"""import os
import json

# Define paths
base_path = os.getcwd()
folder_path = os.path.join(base_path, "patient_cases")
output_file = os.path.join(base_path, "patient_cases_processed.json")

print(f"Scanning directory: {folder_path}")

processed_cases = []

def process_patient_case(file_path, file_name, disease_group, specific_disease=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {file_path}")
            return None

    inputs = {}
    reasoning = {}
    
    # Separate inputs and reasoning
    for key, value in data.items():
        if key.lower().startswith("input"):
            inputs[key] = value
        else:
            reasoning[key] = value
            
    # Clean inputs
    cleaned_inputs = {}
    # Standardize to input1..input6
    for i in range(1, 7):
        key = f"input{i}"
        val = None
        for k in inputs:
            if k.lower() == key:
                val = inputs[k]
                break
        
        if not val or (isinstance(val, str) and val.strip() == ""):
            val = "NA"
        
        if isinstance(val, str):
            cleaned_inputs[key] = val.strip()
        else:
            cleaned_inputs[key] = val

    case_entry = {
        "file_name": file_name,
        "disease_group": disease_group,
        "specific_disease": specific_disease if specific_disease else "NA",
        "reasoning": reasoning,
        "inputs": cleaned_inputs
    }
    
    return case_entry

# Walk through the directory
if not os.path.exists(folder_path):
    print(f"Error: Directory not found: {folder_path}")
else:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                rel_path = os.path.relpath(root, folder_path)
                path_parts = rel_path.split(os.sep)
                
                # path_parts[0] should be 'Finished'
                if len(path_parts) > 1 and path_parts[0] == "Finished":
                    disease_group = path_parts[1]
                    specific_disease = path_parts[2] if len(path_parts) > 2 else None
                    
                    case_data = process_patient_case(file_path, file, disease_group, specific_disease)
                    if case_data:
                        processed_cases.append(case_data)

    # Save the processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_cases, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(processed_cases)} cases. Saved to {output_file}")
"""

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update the 3rd cell (index 2)
if len(nb["cells"]) >= 3:
    nb["cells"][2]["source"] = new_code.splitlines(keepends=True)
else:
    # Append if not enough cells
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_code.splitlines(keepends=True)
    })

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
