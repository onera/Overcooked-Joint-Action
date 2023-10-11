This is a fork from [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) at date of September 2021 followed by a merge of [overcooked-demo](https://github.com/HumanCompatibleAI/overcooked-demo). For internal legal reasons, git history unfortunately cannot be shown.

# Installation
- Clone repo
- Create dedicated virtual environment and activate it, python 3.9.2
- Update pip : `python -m pip install --upgrade pip`
- `cd {install_dir}`
- `pip install -r requirement.txt`

# Run in development mode
- `export FLASK_ENV=development`
- `python app.py`

# Run in production mode
- `export FLASK_ENV=production`
- `python app.py`



# Run an experiment
- Access http://localhost:5000/<config_key>

# Development
## Configurations
Configurations can be found in [config](config.json). New configurations can be generated based on existing ones

## Missions
Missions file can be created in [layouts](overcooked_ai_py/data/layouts). New missions can be generated based on existing ones

## Generate pickle files used by planning engines:
If your agent uses the Medium Leval Action Manager and the Motion planner, you need to pre-compute them :
 - `python compute_mlam.py <config_key_for_which_you_want_to_generate_pickles>`