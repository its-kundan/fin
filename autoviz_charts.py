import shutil
from typing import List, Tuple
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from autoviz.AutoViz_Class import AutoViz_Class # Assuming AutoViz is installed

# Define the absolute target directory for the final charts
# NOTE: This path is specific to your local machine and should be adjusted for portability.
TARGET_CHART_DIR = r"C:\Users\Swarna\Desktop\NVIDIA_agenticAI\fin\charts"


class AutovizChartGenerator:
    """Generate additional charts using AutoViz library for comprehensive visualization"""
    
    def __init__(self, autoviz_dir: str = "autoviz", duplicate_threshold: float = 0.5):
        self.autoviz_dir = autoviz_dir
        self.duplicate_threshold = duplicate_threshold
        self.target_plot_types = ["Bar_Plots", "Dist_Plots_cats", "Dist_plots_numeric", 
                                 "Heat_Maps", "Violin_Plots"]
        
    def _copy_to_target_dir(self, chart_paths: List[str], target_dir: str = TARGET_CHART_DIR) -> None:
        """
        Copies specific chart files (Bar, Dist_Cats, Dist_Numeric) to a designated absolute path.
        """
        try:
            # 1. Create the target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # 2. Define the specific files we want to copy
            specific_files_to_copy = [
                "Bar_Plots.png",
                "Dist_Plots_Cats.png",
                "Dist_Plots_Numeric.png"
            ]
            
            copied_count = 0
            
            # 3. Iterate through all generated chart paths
            for src_path in chart_paths:
                # Check if the filename (basename) matches one of the specific files
                filename = os.path.basename(src_path)
                
                if filename in specific_files_to_copy:
                    # Construct the destination path
                    dst_path = os.path.join(target_dir, filename)
                    
                    # Copy the file
                    shutil.copy(src_path, dst_path)
                    copied_count += 1
            
            print(f" Copied {copied_count} specific charts (Bar, Dist_Cats, Dist_Num) to: {target_dir}")
            
        except Exception as e:
            print(f"  Failed to copy charts to '{target_dir}': {e}")


    def generate_autoviz_charts(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> List[str]:
        """Generate AutoViz charts and return list of chart paths"""
        try:
            # Imports are typically better at the top of the file, but left here as in original code
            from autoviz.AutoViz_Class import AutoViz_Class
            import matplotlib.pyplot as plt
            import glob
            
            print(f" Generating AutoViz charts for {dataset_name}...")
            
            # 1. Filter columns
            cats, nums = self._filter_by_duplicates(df)
            df_filtered = df[cats + nums].copy()
            for col in cats:
                df_filtered[col] = df_filtered[col].astype("category")
            
            if df_filtered.empty:
                print(" No suitable columns found for AutoViz after filtering")
                return []
            
            print(f" Filtered columns: {len(cats)} categorical, {len(nums)} numeric")
            
            # 2. Clear previous outputs
            if os.path.exists(self.autoviz_dir):
                shutil.rmtree(self.autoviz_dir)
            os.makedirs(self.autoviz_dir, exist_ok=True)
            
            # 3. Run AutoViz
            AV = AutoViz_Class()
            AV.AutoViz(
                filename="",
                dfte=df_filtered,
                depVar="",
                chart_format="png",
                verbose=2,
                max_rows_analyzed=min(len(df_filtered), 5000),
                max_cols_analyzed=min(len(df_filtered.columns), 15),
                save_plot_dir=self.autoviz_dir
            )
            plt.close('all')
            
            # 4. Recursively find and copy selected plots to self.autoviz_dir
            chart_paths = []
            for root, _, files in os.walk(self.autoviz_dir):
                for fname in files:
                    if fname.lower().endswith(".png") and any(tp.lower() in fname.lower() for tp in self.target_plot_types):
                        src = os.path.join(root, fname)
                        dst = os.path.join(self.autoviz_dir, fname)
                        if src != dst:
                            shutil.copy(src, dst)
                        chart_paths.append(dst)
            
            print(f" AutoViz generated {len(chart_paths)} selected charts in '{self.autoviz_dir}'")

            # 5. NEW STEP: Copy specific charts to the target user directory
            self._copy_to_target_dir(chart_paths)
            
            return chart_paths
        
        except ImportError:
            print("  AutoViz not installed. Skipping AutoViz chart generation.")
            return []
        except Exception as e:
            print(f"  AutoViz chart generation failed: {e}")
            return []
    
    def _filter_by_duplicates(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Filter columns by duplicate ratio to avoid redundant charts"""
        n = len(df)
        cats, nums = [], []
        for col in df.columns:
            # Using nunique(dropna=True) is correct for counting unique non-NaN values
            dup_ratio = (n - df[col].nunique(dropna=True)) / n
            if dup_ratio > self.duplicate_threshold:
                if pd.api.types.is_numeric_dtype(df[col]):
                    nums.append(col)
                else:
                    cats.append(col)
        return cats, nums
    
    def cleanup(self):
        """Clean up AutoViz temporary files"""
        try:
            if os.path.exists(self.autoviz_dir):
                shutil.rmtree(self.autoviz_dir)
                print(f" Removed AutoViz directory: {self.autoviz_dir}")
        except Exception as e:
            print(f" Cleanup failed: {e}")