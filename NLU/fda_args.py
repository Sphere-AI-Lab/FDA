import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--name", type=str, default='init_by_gauss_random', help="init FDA")
    parser.add_argument("--model", type=str, help="name of the language model", choices=["roberta-base","roberta-large"])
    parser.add_argument("--init_params", type=str, default='pretrained')

    # file location
    parser.add_argument("--data_location", type=str, default='/data/shikexuan/data_cv',help="Root directory for datasets")
    parser.add_argument("--model_location", type=str, default='/data/shikexuan/nlu_finetuned_models', help="Checkpoint directory")
    parser.add_argument("--read_anchors_path", type=str,default='/data/shikexuan/dual_anchors_5880',help='Read anchor path')

    # Anchor
    parser.add_argument("--anchor_num", type=int, default=64, help="Number of anchors")
    parser.add_argument("--token_num", type=int, default=5)
    parser.add_argument("--anchor_loss", type=str, default='cos', choices=['cos', 'mse', 'l1'], help="Loss for anchor generation")
    parser.add_argument("--opt_steps", type=int, default=800, help="Optimization steps for anchor generation")
    parser.add_argument("--opt_lr", type=float, default=1e-2, help="Learning rate for anchor optimization")
    parser.add_argument("--scale", type=float, default=1, help="Scaling factor for Gauss")

    # Adaptation
    parser.add_argument("--adapt_batch_size", type=int, default=128, help="Batch size for adaptation")
    parser.add_argument("--adapt_epochs", type=int, default=500, help="Epochs for adaptation")
    parser.add_argument("--adapt_lr", type=float, default=1e-2, help="Learning rate for adaptation")
    parser.add_argument("--adapt_loss", type=str, default='mse', help="Loss function for adaptation")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha coefficient for adaptation")

    # Saving
    parser.add_argument("--save_model", action="store_true", help="Save model after training")
    parser.add_argument("--save_dual_anchors", action="store_true", help="Save dual anchors")
    parser.add_argument("--save_anchors_path", type=str, default='/data/shikexuan/dual_anchors_5880',help="Anchor save path")
    parser.add_argument("--save_merged_model_path", type=str,default='/data/shikexuan/merged_models')

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args
