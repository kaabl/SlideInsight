## Slide Insight

This repository contains a collection of notebooks for gaining insights into presentation slides through multimodal AI models. Some goals are:

- [Comparison of different models](Test_Models.ipynb) and their performance on summarizing the content of presentation slides. This is not implemented through text-to-text models but rather through image-to-text (multimodal) models. As a first test pdf I3D:bio's Training Material ['WhatIsOMERO.pdf'](https://doi.org/10.5281/zenodo.8323588) (Schmidt, C., Bortolomeazzi, M. et al., 2023) is used.
- Establishing a workflow that is capable of [grouping Slides](Text_Embedding.ipynb) from multiple presentations together based on their representation as a word embedding. This helps to gather all available information concerning one specific topic from different presentations. Testing Data is again I3D:bio's Training Material and also Presentation Slides from the [Bio-image Data Science Lectures](https://zenodo.org/records/12623730) from Robert Haase (licensed under CC-BY 4.0).
- Improve our understanding about [how different types of embeddings represent the same content](Compare_Embeddings.ipynb). For this task, some presentation slides are adapted to see whether text, visual or mixed-modal embeddings perform comparably well in representating a slides features, when the slide is changed in a specific manner. For this, slides are adapted from the Bio-image Data Science Lectures.

To access the AI models used in this repository, this [free Service from Github](https://github.com/marketplace/models) is used.

***Be aware that there are certain [rate limits](https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits) for each model!***



### Before getting started:
Make sure to generate a developer key / personal access token on Github and set it as an environment variable. You can generate the token via the [Github website](github.com) under user settings and afterwards set it like this for your current session:


##### bash:
```export GITHUB_TOKEN= "your-github-token-goes-here"```

##### powershell:
```$Env:GITHUB_TOKEN= "your-github-token-goes-here"```

##### Windows command prompt:
```set GITHUB_TOKEN= your-github-token-goes-here```

