#Install
python -m venv venv4
venv\Scripts\Activate
python -m pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

# Build the .whl
python -m venv venv4
venv\Scripts\Activate
python -m pip install --upgrade pip
pip install wheel
pip install build
python -m build
