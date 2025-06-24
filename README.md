# Generative machine learning models for music creation

This repository contains the code for my undergraduate thesis **“Generative machine learning models for music creation”**. It builds heavily on the [NotaGen](https://github.com/ElectricAlexis/NotaGen) symbolic music generation model; many thanks to the NotaGen authors for making their code available.

## Thesis Abstract

The rapid advancement of machine learning models, especially large language models, over the past few years has yielded unprecedented capabilities in generating both text and code. An increasing number of systems are now being developed to emulate creative tasks such as painting, image processing, and music composition.

Although today’s generative music research for the most part focuses on audio outputs, the aim of this thesis is to utilize the success of language models for the creation of symbolic music. This becomes possible thanks to the strong parallels between natural language and musical symbol sequences.

Building on the pre-trained NotaGen model, which has been trained on music in textual form, this study applies transfer learning techniques to adapt (finetune) the model to a new dataset of classical guitar works. We perform a comparative evaluation of the methods tackling this challenge and introduce an evaluation function designed to judge whether the final outputs are playable by human performers.

## Dataset

This study uses the **Classical Guitar MIDI Dataset**, a collection of over 5.000 MIDI files of classical guitar works spanning Baroque to Romantic eras, including 475 files featuring contra guitar and a variety of other plucked, bowed, wind, and percussion instruments. The dataset is publicly available at [https://www.classicalguitarmidi.com/](https://www.classicalguitarmidi.com/) (© 1998–2024 François Faucher).

## Main Findings

In our experiments, the **Finetune All** approach significantly outperformed other adaptation strategies on key quality and capability metrics—achieving the lowest values in perplexity, CLaMP-2 score and Frechet Music Distance. The **Finetune Last** method yielded similar yet slightly inferior results, indicating that the character-level decoder drives most of the sequence modeling, while the patch-level decoder provides marginal but meaningful improvements. Baseline (non-transfer) and zero-shot methods lagged behind across all metrics, underscoring the necessity of transfer learning for adapting NotaGen to classical guitar repertoire.

The results of the playability metric demonstrate that, on average, the generated outputs closely approach the playability of human compositions. All adaptation methods achieve playability scores within the second decimal place of the human baseline, with negligible differences between them :contentReference[oaicite:3]{index=3}.


## Generated Outputs

Please explore the musical examples produced by the fine-tuned model in the `data/generations` directory to hear and inspect the results.

**Note:** The audio renderings of these pieces were generated using MuseScore; as such, the instrument sounds may not reflect professional-quality timbres.


## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.