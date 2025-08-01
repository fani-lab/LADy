import os
import pandas as pd

def collect_excel_files(base_folder):
    #Recursively collects all .xlsx files under the base folder.
    excel_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".xlsx"): excel_files.append(os.path.join(root, file))
    return excel_files

def group_by_top_folder(files, base_folder):
    #Groups file paths by their top-level folder within the base folder.
    grouped = {}
    for f in files:
        parts = os.path.relpath(f, base_folder).split(os.sep)
        group = parts[0]  # top-level directory
        grouped.setdefault(group, []).append(f)
    return grouped

def summarize_files(file_list):
    #Extracts and combines 'Evaluation_summary' sheets from a list of Excel files.
    summaries = {}
    for path in file_list:
        try:
            summary_df = pd.read_excel(path, sheet_name="Evaluation_summary")
            summary_df.insert(0, "File_Name", os.path.basename(path))  # Add filename to each row
            summaries[os.path.basename(path)] = summary_df
        except Exception as e: print(f"Skipping {path}: {e}")
    if summaries: return pd.concat(summaries.values(), ignore_index=True)
    else: return pd.DataFrame()



def generate_overall_metrics(summary_file_path):
    #Reads a summary Excel file and computes macro/micro accuracy metrics.
    #Adds a new sheet named 'Overall_Metrics' to the same file.
    df = pd.read_excel(summary_file_path, sheet_name=None)
    
    if 'Sheet1' in df: data = df['Sheet1']
    else: data = next(iter(df.values()))

    macro_accuracy = data["Accuracy"].mean()
    total_correct = data["Correct Predictions"].sum()
    total_cases = data["Total Cases"].sum()
    micro_accuracy = (total_correct / total_cases) * 100 if total_cases > 0 else 0

    overall_analysis = pd.Series({
        "Macro Accuracy (%)": round(macro_accuracy, 2),
        "Micro Accuracy (%)": round(micro_accuracy, 2),
        "Total Files": len(data),
        "Total Cases": total_cases,
        "Total Correct Predictions": total_correct,
        "Total Incorrect Predictions": data["Incorrect Predictions"].sum()
    })

    # Save back to the same Excel file with a new sheet
    with pd.ExcelWriter(summary_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer: overall_analysis.to_frame(name='Value').to_excel(writer, sheet_name='Overall_Metrics')

    print(f"Overall metrics added to sheet 'Overall_Metrics' in {summary_file_path}")

def main(base_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)  # Ensure the save folder exists

    all_files = collect_excel_files(base_folder)
    grouped_files = group_by_top_folder(all_files, base_folder)
    all_group_summaries = []
    for group, files in grouped_files.items():
        print(f"Processing group: {group} with {len(files)} files")
        group_summary = summarize_files(files)
        if not group_summary.empty:
            group_summary.insert(0, "Dataset_Group", group)
            group_summary.to_excel(os.path.join(save_folder, f"{group}_trustworthiness_summary.xlsx"), index=False)
            all_group_summaries.append(group_summary)

    if all_group_summaries:
        final_summary = pd.concat(all_group_summaries, ignore_index=True)
        file_path = os.path.join(save_folder, "overall_trustworthiness_summary.xlsx")
        final_summary.to_excel(file_path, index=False)
        generate_overall_metrics(file_path)
        print("All summaries generated successfully.")
    else: print("No valid summaries found.")


main("/Users/karanveersinghsidhu/LADy_Model_Eval_Results", "/Users/karanveersinghsidhu/LADy_Model_Eval_Results")
