# RAVEN

### Notice
- We have rewrite our dataloader for MOSI in new CMU-Multimodal SDK (see https://github.com/A2Zadeh/CMU-MultimodalSDK). However, we haven't found the best hyperparameters on this new kind of data. And it seems to have some bugs in the dataloader. We are trying to solve that. Thank you for your patiences!

### File description

- `main.py` : the entrance to the program. You can simply run it to get a quick start.
- `model.py` : where the model is defined. 
- `layer.py` : an implementation of simple LSTM, it should work the same as `torch.nn.LSTM` does. 
- `consts.py` : where all adjustable parameters and shared variables store. 
- `ie_data_loader.py` : data loader for IEMOCAP.
- `ie_dataset.py` : derived from `torch.dataset`, a class for feeding IEMOCAP data.
- `MOSI_dataset.py`:derived from `torch.dataset`, a class for feeding CMU-MOSI data.

### Usage

#### Quick run:

```
python main.py
```

