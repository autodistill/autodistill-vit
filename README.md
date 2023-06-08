<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill ViT Module

This repository contains the code supporting the ViT target model for use with [Autodistill](https://github.com/autodistill/autodistill).

[ViT](https://huggingface.co/google/vit-base-patch16-224-in21k) is a classification model pre-trained on ImageNet-21k, developed by Google. You can train ViT classification models using Autodistill.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [ViT Autodistill documentation](https://autodistill.github.io/autodistill/target_models/vit/).

## Installation

To use the ViT target model, you will need to install the following dependency:

```bash
pip3 install autodistill-vit
```

## Quickstart

```python
from autodistill_vit import ViT

target_model = ViT()

# train a model from a classification folder structure
target_model.train("./context_images_labeled/", epochs=200)

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg", conf=0.01)
```

## License

The code in this repository is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!