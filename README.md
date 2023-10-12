# Overcooked for Human/Artificial Agent Joint Action study

Developed as part of a thesis using a software base developed by the BAIR team (https://bair.berkeley.edu/) at UC Berkeley. It is a fork of overcooked_ai (https://github.com/HumanCompatibleAI/overcooked_ai) merged with Overcooked-demo (https://github.com/HumanCompatibleAI/overcooked-demo). The modifications made at ONERA make it possible to use a rule-based agent and to display different levels of information about its operation in the interface, in order to measure the impact of this information on the human-artificial agent interaction from a behavioural and subjective point of view. The modifications made at ONERA mean that a configuration file can be used to deploy studies that vary the missions proposed to participants, and to propose questionnaires at the end of mission blocks or the missions themselves. This software is intended to be available internally and republished as open-source. It was used to generate data for a paper (DAP DTIS23007) which will be presented at AAMAS23 (https://aamas2023.soton.ac.uk/).

## Installation
- Clone repo
- Create dedicated virtual environment and activate it, python 3.9.2
- Update pip : `python -m pip install --upgrade pip`
- `cd {install_dir}`
- `pip install -r requirement.txt`

## Run in development mode
- `export FLASK_ENV=development`
- `python app.py`

## Run in production mode
- `export FLASK_ENV=production`
- `python app.py`



## Run an experiment
- Access http://localhost:5000/<config_key>

## Development
### Configurations
Configurations can be found in [config](config.json). New configurations can be generated based on existing ones

### Missions
Missions file can be created in [layouts](overcooked_ai_py/data/layouts). New missions can be generated based on existing ones

### Generate pickle files used by planning engines:
If your agent uses the Medium Leval Action Manager and the Motion planner, you need to pre-compute them :
 - `python compute_mlam.py <config_key_for_which_you_want_to_generate_pickles>`

## License

MIT License
Copyright (c) ONERA