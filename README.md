# README

## Role Mapping – Batch AI Classifier

This project performs large-scale commercial/non-commercial role classification using a batched OpenAI pipeline and a multi-sheet taxonomy workbook. It is designed to run locally inside a Linux environment (Windows Subsystem for Linux recommended) using Python virtual environments.

---

# 1. Install Linux Subsystem (WSL) and Set Up VS Code

### Windows users
1. Open PowerShell as Administrator
2. Enable WSL:
   wsl --install
3. Restart your machine when prompted
4. Open VS Code → install the Remote Development extension pack
5. In VS Code, open the WSL Linux environment:
   - Ctrl + Shift + P → 'WSL: New WSL Window'
   - Select your Linux distro (Ubuntu recommended)

---

# 2. Create & Activate Python Virtual Environment

cd /path/to/your/project

python3 -m venv venv

source venv/bin/activate

(You should now see (venv) in your terminal.)

To deactivate:

deactivate

---

# 3. Install Dependencies

pip install -r requirements.txt

(Optional) Upgrade pip:

pip install --upgrade pip

---

# 4. Add Your OpenAI API Key

Create a .env file:

echo "OPENAI_API_KEY=\"<YOUR_API_KEY>\"" > .env

---

# 5. Run the Mapping Script

python main.py

Default inputs:
- input_census.csv
- taxonomy_updated.xlsx

Default output:
- mapped_census.csv

Override paths using env vars:
CENSUS_PATH="..."
TAXONOMY_PATH="..."
OUTPUT="..."

---

# 6. Project Structure

project/
├── main.py
├── batch_mapping.py
├── org_paths.py
├── Hierarchy_reclassification.py
├── taxonomy_updated.xlsx
├── input_census.csv
├── requirements.txt
├── .env
└── README.md

---

# 7. Support

Check these if issues occur:
1. venv is active
2. .env contains valid API key
3. requirements.txt installed correctly

