# README

# Role Mapping – Batch AI Classifier (Commercial + Non-Commercial)

This repo maps employee job titles into a **two-stage taxonomy**:

- **Stage 1 (L1):** Commercial vs non-commercial routing + commercial L1 bucket selection  
- **Stage 2 (L2):** Commercial L2 selection constrained by the chosen L1 (or keep BU/Team for non-commercial)

It’s built for **large-scale batches** using OpenAI (batched prompts) plus a **taxonomy workbook** (L1/L2 + example titles + optional deterministic rules).

> Primary intent: **Commercial role mapping** (Sales/Marketing/CS/Support/Pricing/RevOps/PM) while **preserving non-commercial functions** (Finance/HR/IT/etc.) as-is.

---

## How it works (high level)

### Inputs
- **Employee census file** (CSV / XLSM export)
  - Key columns typically include: `Job-Profile` (title), `Team` (or BU), optional `Manager ID`, etc.
- **Taxonomy workbook** (Excel)
  - Must include `L1` and `L2`
  - Includes example titles columns (e.g., `Final Titles`, `Example Titles (Additional)`)
  - Optional “rules” sheet with regex/contains/equals patterns mapped to L1/L2

### Stage 1: L1 Mapping (Commercial-first)
For each employee:
1. Build signals from:
   - Job title (primary)
   - BU/Team (secondary, weighted)
   - Anchor patterns (must/contra) for commercial buckets
   - Optional rule hints (from taxonomy rules)
2. **Special handling for commercial C-suite titles**:
   - Only C-suite titles that exist in the taxonomy example titles are mapped into the corresponding commercial L1.
   - Non-commercial C-suite titles (e.g., CEO/CFO/CISO/People/Risk) remain non-commercial unless taxonomy explicitly classifies them as commercial.
3. Output:
   - `Mapped_L1` (one of the canonical commercial 7, or “keep BU/Team” for non-commercial)
   - `L1_Confidence`
   - `L1_Source` (audit trail)

### Stage 2: L2 Mapping (Constrained by L1)
For each employee:
1. If `Final_Mapped_L1` is **non-commercial** → keep BU/Team as L2
2. If `Final_Mapped_L1` is **commercial**:
   - Allowed L2 list is `taxonomy[Final_Mapped_L1]`
   - GPT must select **exactly one** L2 from that allowed list
3. Leadership enforcement:
   - “Leadership” L2 is only allowed if the title meets a strict **true leadership** definition (C-suite / VP+ / Director / Head / President)
   - Managers/leads are explicitly excluded
   - A separate `Manager_Title_Flag` is set for visibility
4. Output:
   - `Mapped_L2`
   - `L2_Confidence`
   - `L2_Source`

# 1. Install Linux Subsystem (WSL) and Set Up VS Code (Optional)

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

# 2. Create & Activate Python Virtual Environment (Linux)

cd /path/to/your/project

python3 -m venv venv

source venv/bin/activate

(You should now see (venv) in your terminal.)

To deactivate:

deactivate

---

# 2. Create & Activate Python Virtual Environment (Windows)

cd /path/to/your/project

pip install virtualenv

python -m venv venv

venv\Scripts\activate

# 3. Install Dependencies

pip install -r requirements.txt

(Optional) Upgrade pip:

pip install --upgrade pip

---

# 4. Add Your OpenAI API Key

Create a .env file:

echo "OPENAI_API_KEY=\"<YOUR_API_KEY>\"" > .env (Linux)

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

