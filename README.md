# LLM_RecSys: Large Language Model–based Recommendation System

This repository contains experimental code and notebooks for **accuracy, fairness, and Diversity Evaluations**
It explores how prompting formats, decoding temperatures, and decoding strategies affect recommendation quality across models such as **OpenAI GPT-4o** and **Mistral** families.

---

## Project Structure

| File / Folder | Description |
|----------------|-------------|
| `gpt_recommendation.ipynb` | Notebook for running GPT-based recommendation generation for MovieLens. |
| `Mistral_Large-2.ipynb` | Notebook for running Mistral-based recommendation generation MovieLens. |
| `greedy_template_analysis.ipynb` | Notebook for extracting recommendations across different prompt formats and analyzing their performance MovieLens. |
| `temperature_analysis.ipynb` | Notebook for conducting temperature analysis across models MovieLens. |
| `sc_results.ipynb` | Notebook for analyzing self-consistency decoding results MovieLens. |
| `data/final_movies.csv` | movie details from movie lens - 100k dataset |
| `data/movies.csv` | movie details from 32M and some verified and added manually for real-world movie verification |
| `Steam_GPT.ipynb` | Notebook for running GPT-based recommendation generation for steam. |
| `Steam_Mistral.ipynb` | Notebook for running Mistral-based recommendation generation for steam. |
| `steam_data_prep.py` | data preparation for recommendation generation in steam. |
| | `data/steam-200k.csv` | game data |
