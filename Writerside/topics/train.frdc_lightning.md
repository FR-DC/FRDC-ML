# train.frdc_datamodule & frdc_module

<tldr>
The FRDC PyTorch LightningDataModule and LightningModule.
</tldr>

These are FRDC specific LightningDataModule and LightningModule,
a core component in the PyTorch Lightning ecosystem to provide a simple
interface to train and evaluate models.

## Classes

> It's optional to use these classes, you can use your own training loop
> if you want. We'll use these for our training pipeline.
> {style='note'}

<deflist type="medium">
<def title="FRDCDataModule">
<b>The FRDC PyTorch Lightning DataModule.</b>
</def>
<def title="FRDCModule">
<b>The FRDC PyTorch Lightning Module.</b>
</def>
</deflist>

## Usage

> See our training pipeline for a full example

## API

<deflist>
<def title="FRDCDataModule(segments, labels, preprocess, augmentation, train_val_test_split, batch_size)">
<b>Initializes the FRDC PyTorch Lightning DataModule.</b><br/>
<list>
<li><code>segments</code>, <code>labels</code> are retrieved from
<list>
<li><a href="load.dataset.md">FRDCDataset</a></li>
<li><a href="preprocessing.extract_segments.md">Segmentation</a></li>
</list>
</li>
<li><code>preprocess</code> is a function that takes in a segment and returns a preprocessed
segment. In particular, it should accept a list of NumPy NDArrays and return
a single stacked PyToch Tensor.</li>
<li><code>augmentation</code> is a function that takes in a segment and returns an augmented
segment. In particular, it takes in a PyTorch Tensor and returns another.</li>
<li><code>train_val_test_split</code> is a function that takes a TensorDataset and returns
a list of 3 TensorDatasets, for train, val and test respectively.</li>
<li><code>batch_size</code> is the batch size.</li>
</list>
<note>For now, the <code>augmentation</code> is only applied to training
data</note>
</def>
<def title="FRDCModule(model_cls, model_kwargs, optim_cls, optim_kwargs)">
<b>Initializes the FRDC PyTorch Lightning Module.</b><br/>
<list>
<li><code>model_cls</code> is the Class of the model.</li>
<li><code>model_kwargs</code> is the kwargs to pass to the model.</li>
<li><code>optim_cls</code> is the Class of the optimizer.</li>
<li><code>optim_kwargs</code> is the kwargs to pass to the optimizer.</li>
</list>
Internally, the module will initialize the model and optimizer as follows:
<code-block lang="python">
model = model_cls(**model_kwargs)
optim = optim_cls(model.parameters(), **optim_kwargs)
</code-block>
<note>We do not accept the instances of the Model and Optimizer so
that we can pickle them.</note>
</def>
</deflist>
