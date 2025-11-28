import json
import os

def process_data(input_file, output_file):
    """
    Processes the combined_rag_data.json file into a format suitable for RAG.
    Separates Medical Knowledge Base (KB) and Patient Cases.
    """
    print(f"Loading data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    knowledge_base = []
    patient_cases = []

    print(f"Processing {len(data)} items...")

    for item in data:
        # Check for Knowledge Base Item
        if "medicalKB" in item:
            # Format: "type/subtype/...: content" or just "content"
            # We'll store the raw string as 'content' and try to extract a 'topic' if possible
            kb_text = item["medicalKB"]
            
            # Simple parsing to separate topic from content if a colon exists
            if ": " in kb_text:
                parts = kb_text.split(": ", 1)
                topic = parts[0]
                content = parts[1]
            else:
                topic = "General"
                content = kb_text

            kb_item = {
                "id": item.get("id", f"KB_{len(knowledge_base)}"),
                "topic": topic,
                "content": content,
                "raw_text": kb_text
            }
            knowledge_base.append(kb_item)

        # Check for Patient Case
        elif "patient_case" in item:
            case_data = item["patient_case"]
            case_id = item.get("id", f"CASE_{len(patient_cases)}")
            
            # Extract inputs (clinical notes)
            inputs = case_data.get("inputs", {})
            # Combine all input text into one context string for the model
            full_input_text = ""
            for key, value in inputs.items():
                if value and value != "None":
                    full_input_text += f"{key}: {value}\n"
            
            # Extract reasoning and diagnosis
            reasoning_chain = case_data.get("reasoning", "")
            disease_group = case_data.get("disease_group", "Unknown")
            specific_disease = case_data.get("specific_disease", "Unknown")

            case_item = {
                "id": case_id,
                "disease_group": disease_group,
                "specific_disease": specific_disease,
                "input_text": full_input_text.strip(),
                "gold_reasoning": reasoning_chain,
                "gold_diagnosis": specific_disease
            }
            patient_cases.append(case_item)

    # Construct final dataset
    processed_data = {
        "knowledge_base": knowledge_base,
        "cases": patient_cases
    }

    print(f"Extracted {len(knowledge_base)} KB items and {len(patient_cases)} Patient Cases.")
    
    print(f"Saving processed data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust path to where the file is located based on previous `list_dir`
    # It was in: mimic-iv-ext-direct-1.0.0/combined_rag_data.json
    input_path = os.path.join(base_dir, "mimic-iv-ext-direct-1.0.0", "combined_rag_data.json")
    output_path = os.path.join(base_dir, "processed_rag_dataset.json")

    process_data(input_path, output_path)
