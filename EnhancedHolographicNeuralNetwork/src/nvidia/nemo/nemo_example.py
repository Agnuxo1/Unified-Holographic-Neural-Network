import nemo
import nemo.collections.asr as nemo_asr

# Load a pre-trained ASR model
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Transcribe audio
files = ['path/to/audio_file.wav']
transcriptions = asr_model.transcribe(paths2audio_files=files)

print(transcriptions)

# Fine-tune the model
train_data = 'path/to/train_manifest.json'
validation_data = 'path/to/val_manifest.json'

asr_model.setup_training_data(train_data_config={
    'manifest_filepath': train_data,
    'labels': asr_model.decoder.vocabulary,
    'batch_size': 32,
    'shuffle': True,
})

asr_model.setup_validation_data(val_data_config={
    'manifest_filepath': validation_data,
    'labels': asr_model.decoder.vocabulary,
    'batch_size': 32,
    'shuffle': False,
})

trainer = nemo.core.PyTorchLightning.Trainer(max_epochs=50, gpus=1)
trainer.fit(asr_model)

# Save the fine-tuned model
asr_model.save_to('path/to/save/model.nemo')

