"""
Master Pipeline Runner: Cognifyz Data Science Internship Project
===============================================================
Executes all Level 1, Level 2, and Level 3 tasks sequentially.
"""

import os
import sys
import subprocess
import time

def run_step(script_relative_path):
    print("\n" + "=" * 75)
    print(f"RUNNING: {script_relative_path}")
    print("=" * 75)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, script_relative_path),
        os.path.join(base_dir, "..", script_relative_path)
    ]
    script_path = None
    for p in candidates:
        if os.path.exists(p):
            script_path = p
            break
            
    if not script_path:
        raise FileNotFoundError(f"Could not locate {script_relative_path}")
        
    start_time = time.time()
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"[ERROR] Failed with code {result.returncode}:")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print(result.stdout)
        print(f"[SUCCESS] Completed in {duration:.2f} seconds.")

def main():
    print("***************************************************************************")
    print("  COGNIFYZ TECHNOLOGIES: RESTAURANT DATA ANALYTICS & ML PIPELINE RUNNER   ")
    print("***************************************************************************")
    
    tasks = [
        "level_1/task_1_preprocessing.py",
        "level_1/task_2_descriptive_analysis.py",
        "level_1/task_3_geospatial_analysis.py",
        "level_2/task_1_booking_delivery.py",
        "level_2/task_2_price_analysis.py",
        "level_2/task_3_feature_engineering.py",
        "level_3/task_1_predictive_modeling.py",
        "level_3/task_2_customer_preferences.py",
        "level_3/task_3_visualization.py"
    ]
    
    total_start = time.time()
    for task in tasks:
        run_step(task)
        
    total_duration = time.time() - total_start
    print("\n" + "*" * 75)
    print(f"  ALL TASKS COMPLETED SUCCESSFULLY IN {total_duration:.2f} SECONDS!  ")
    print("***************************************************************************")
    print("\nTo launch the interactive Streamlit Dashboard, run:")
    print("    streamlit run app.py\n")

if __name__ == "__main__":
    main()
