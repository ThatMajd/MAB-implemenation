# MAB Implemenation 🎰

## Overview

This repository contains code for an assignment in the course [e-commerce](https://students.technion.ac.il/local/technionsearch/course/96211). The system utilizes a Multi-Armed Bandit (MAB) approach to optimize the selection of content providers for users. 

## Dependencies

The following Python libraries are required to run the code:

- numpy

- pandas

- networkx

- matplotlib

# Repository Structe
The repository includes 3 files:
- [model.py](https://github.com/ThatMajd/MAB-implemenation/blob/main/model.py) - The code for the model
- [simulation.py](https://github.com/ThatMajd/MAB-implemenation/blob/main/simulation.py) - the enviorment that the model runs in.
- [report](https://github.com/ThatMajd/MAB-implemenation/blob/main/206528382_323958140.pdf) - A report detaling how the model works .

## Basic Walkthrough
The model can be broken to 2 stages:
- **Explore** the model samples from the enviorment by choosing random arms, while also mainitaing their thresholds so we won't lose any arms, once the model thinks it has a pretty accurate estimation for the enviorment
the model runs an enternal "Fake simulation" that runs almost like the real world using the parameters it learned to find the optimal subset of arms to play in order to maxmize it's reward.

- **Exploit** the model will then exploit it's knowledge of the best arms to maximize the reward


## Conclusion

The code in this repository provides a foundation for building and testing a Multi-Armed Bandit-based content recommendation system. Feel free to adapt and expand the code to suit your specific use case, dataset, and requirements.

For any questions or issues related to the code, please feel free to contact me. 
