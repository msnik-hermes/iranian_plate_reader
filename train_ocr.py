from src.trainer import OCRTrainer
from src.utils import CHAR_TO_ID
from torch.utils.data import DataLoader
from src.ocr import PlateDataset

# دیتاست
train_dataset = PlateDataset(
    images_dir="data/generated_plates/images",
    labels_dir="data/generated_plates/labels"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# آموزش
trainer = OCRTrainer(num_chars=len(CHAR_TO_ID))
trainer.train(train_loader, epochs=50, save_dir="models")

print("Training completed!")


