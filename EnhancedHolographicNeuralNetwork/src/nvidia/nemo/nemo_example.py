import nemo
import nemo.collections.asr as nemo_asr
import os
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HolographicASR:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def setup_model(self):
        try:
            self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                model_name="QuartzNet15x5Base-En"
            ).to(self.device)
            logger.info("Modelo cargado correctamente")
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}")
            raise

    def transcribe(self, audio_files):
        return self.model.transcribe(paths2audio_files=audio_files)

# Load a pre-trained ASR model
asr_model = HolographicASR()
asr_model.setup_model()

# Transcribe audio
files = ['path/to/audio_file.wav']
transcriptions = asr_model.transcribe(audio_files=files)

print(transcriptions)

# Fine-tune the model
train_data = 'path/to/train_manifest.json'
validation_data = 'path/to/val_manifest.json'

asr_model.model.setup_training_data(train_data_config={
    'manifest_filepath': train_data,
    'labels': asr_model.model.decoder.vocabulary,
    'batch_size': 32,
    'shuffle': True,
})

asr_model.model.setup_validation_data(val_data_config={
    'manifest_filepath': validation_data,
    'labels': asr_model.model.decoder.vocabulary,
    'batch_size': 32,
    'shuffle': False,
})

trainer = nemo.core.PyTorchLightning.Trainer(max_epochs=50, gpus=1)
trainer.fit(asr_model.model)

# Save the fine-tuned model
asr_model.model.save_to('path/to/save/model.nemo')

