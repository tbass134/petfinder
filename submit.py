from src.modules import PetFinderModule
import glob

for ckpt in glob.glob("checkpoints/*.ckpt"):
    print(ckpt)
    model = PetFinderModule.load_from_checkpoint(ckpt)
    print(model)
    break
