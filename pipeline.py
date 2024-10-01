from data_preparation import prep_lr
from model import build_degradation_encoder, build_generator
from training import train_models

def run_pipeline():
    # Data Preparation
    prep_lr(
        HR_path='/home/bastin/PROJECT-main/Data/Train/HR/',
        LR_path='/home/bastin/PROJECT-main/Data/Train/LR/',
        HR_patch_path='/home/bastin/PROJECT-main/Data/Train/Patch_HR/',
        LR_patch_path='/home/bastin/PROJECT-main/Data/Train/Patch_LR/',
        SR_scale=4
    )

    # Model Building
    degradation_encoder = build_degradation_encoder()
    generator = build_generator()

    # Training
    train_models(
        degradation_encoder,
        generator,
        train_path='/path/to/train_data',
        test_path='/path/to/test_data'
    )

if __name__ == "__main__":
    run_pipeline()
