import os
import subprocess
import json
import argparse
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description="Evaluate Mean Teacher Checkpoints")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoint_mean_teacher_latest.pth",
                        help="Path to checkpoint file (default: models/checkpoint_mean_teacher_latest.pth)")
    parser.add_argument("--suffix", type=str, default="latest",
                        help="Suffix to append to predictions directories and JSON results (default: latest)")
    args = parser.parse_args()
    
    # Dynamic visible devices for SLURM compatibility
    env = os.environ.copy()
    cuda_dev = os.getenv("CUDA_VISIBLE_DEVICES", "2") # Default to 2 on Jumbito to run on the free Ada GPU
    env["CUDA_VISIBLE_DEVICES"] = cuda_dev
    
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
        
    print(f"====================================================")
    # Print system setup
    print(f"Starting Mean Teacher Validation Evaluation Pipeline")
    print(f"Isolated CUDA GPU: {cuda_dev}")
    print(f"Target Checkpoint: {checkpoint_path}")
    print(f"====================================================\n")
    
    # ----------------------------------------------------
    # 1. Evaluate STUDENT Weights
    # ----------------------------------------------------
    student_pred_dir = f"data/predictions_mean_teacher_student_{args.suffix}"
    student_eval_out = f"data/eval_results_mean_teacher_student_{args.suffix}.json"
    
    print("[1/4] Running batch validation inference using STUDENT weights...")
    os.makedirs(student_pred_dir, exist_ok=True)
    cmd_student_inf = [
        ".venv-voxtell/bin/python",
        "scripts/voxtell/voxtell_inference.py",
        "--split", "val",
        "--output_dir", student_pred_dir,
        "--checkpoint", checkpoint_path,
        "--tile_step_size", "0.5"
    ]
    subprocess.run(cmd_student_inf, check=True, env=env)
    
    print("[2/4] Running metrics evaluation for STUDENT predictions...")
    cmd_student_eval = [
        ".venv-voxtell/bin/python",
        "scripts/evaluate.py",
        "--split", "val",
        "--pred_dir", student_pred_dir,
        "--output_json", student_eval_out
    ]
    subprocess.run(cmd_student_eval, check=True)
    
    # ----------------------------------------------------
    # 2. Evaluate TEACHER Weights
    # ----------------------------------------------------
    teacher_pred_dir = f"data/predictions_mean_teacher_teacher_{args.suffix}"
    teacher_eval_out = f"data/eval_results_mean_teacher_teacher_{args.suffix}.json"
    
    print("\n[3/4] Running batch validation inference using TEACHER weights...")
    os.makedirs(teacher_pred_dir, exist_ok=True)
    cmd_teacher_inf = [
        ".venv-voxtell/bin/python",
        "scripts/voxtell/voxtell_inference.py",
        "--split", "val",
        "--output_dir", teacher_pred_dir,
        "--checkpoint", checkpoint_path,
        "--use_teacher",
        "--tile_step_size", "0.5"
    ]
    subprocess.run(cmd_teacher_inf, check=True, env=env)
    
    print("[4/4] Running metrics evaluation for TEACHER predictions...")
    cmd_teacher_eval = [
        ".venv-voxtell/bin/python",
        "scripts/evaluate.py",
        "--split", "val",
        "--pred_dir", teacher_pred_dir,
        "--output_json", teacher_eval_out
    ]
    subprocess.run(cmd_teacher_eval, check=True)
    
    # ----------------------------------------------------
    # 3. Print Results Summary Comparison
    # ----------------------------------------------------
    # Load baseline
    baseline_path = "data/eval_results_baseline_validation.json"
    baseline_dice = 0.213864
    baseline_hit = 0.486957
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            b_data = json.load(f)
            baseline_dice = b_data.get("average_dice", baseline_dice)
            baseline_hit = b_data.get("hit_rate_0.1", baseline_hit)
            
    # Load Student
    with open(student_eval_out, 'r') as f:
        s_data = json.load(f)
        student_dice = s_data.get("average_dice", 0.0)
        student_hit = s_data.get("hit_rate_0.1", 0.0)
        
    # Load Teacher
    with open(teacher_eval_out, 'r') as f:
        t_data = json.load(f)
        teacher_dice = t_data.get("average_dice", 0.0)
        teacher_hit = t_data.get("hit_rate_0.1", 0.0)
        
    print("\n" + "="*60)
    print("      QUANTITATIVE VALIDATION PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Configuration':<25} | {'Average Dice':<15} | {'Hit Rate (>=0.1)':<15}")
    print("-"*60)
    print(f"{'Zero-Shot Baseline (v1.1)':<25} | {baseline_dice:<15.4f} | {baseline_hit:<15.4%}")
    print(f"{'Mean Teacher Student':<25} | {student_dice:<15.4f} | {student_hit:<15.4%}")
    print(f"{'Mean Teacher Teacher (EMA)':<25} | {teacher_dice:<15.4f} | {teacher_hit:<15.4%}")
    print("="*60)
    
if __name__ == "__main__":
    main()
