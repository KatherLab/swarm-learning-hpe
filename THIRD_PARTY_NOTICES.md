# Third-party components

This repository bundles code from other projects. Each keeps its own licence:

- `workspace/swag-latent-diffusion/model`: MONAI Generative Models (Apache License 2.0; licence file in that folder), https://github.com/Project-MONAI/GenerativeModels
- `workspace/score-based-model/model` (runners, losses, datasets, noise-conditional score networks): derived from ncsnv2 by Yang Song (MIT License, Copyright (c) 2020 Yang Song), https://github.com/ermongroup/ncsnv2
- `workspace/score-based-model/model/models/layers.py` and the UNet modules built on it: derived from denoising-diffusion-pytorch by Kim Seonghyeon (MIT License, Copyright (c) 2020 Kim Seonghyeon), https://github.com/rosinality/denoising-diffusion-pytorch

HPE Swarm Learning is proprietary software of Hewlett Packard Enterprise. Its installers, licence files and documentation are not part of this repository.
