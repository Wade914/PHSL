#!/usr/bin/env python3
"""
TDHNN Multi-Dataset Experiment Launch Script

How to run:
python run_experiment.py

This script will test the TDHNN model on 11 different datasets,
running 5 independent experiments on each dataset and generating detailed performance reports.
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run main experiment
if __name__ == "__main__":
    print("Starting TDHNN multi-dataset experiment...")
    print("=" * 60)
    print("Experiment configuration:")
    print("- Dataset suffixes: 30, 20, 10, 5, 0, -5, -10, -20, -30, -40, -50")
    print("- Model: TDHNN")
    print("- Running 5 independent experiments per dataset")
    print("- Total experiments: 11 × 5 = 55 experiments")
    print("=" * 60)
    
    try:
        # Import and run experiment
        from exp_multi_dataset import *
        print("Experiment starting...")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all necessary modules are correctly installed")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred during experiment: {e}")
        print("Please check if data files exist and model is correctly configured")
        sys.exit(1) 