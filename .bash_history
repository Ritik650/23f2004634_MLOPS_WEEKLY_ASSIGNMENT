git clone -b week_1 https://github.com/IITMBSMLOps/ga_resources.git
pip install -r requirements.txt
pip install -r requirements.txt
pip install -r requirements.txt
python data_prep.py --src ga_resources/data/raw/iris.csv --version raw
python data_prep.py --src ga_resources/data/raw/iris.csv --version raw
python data_prep.py --src ga_resources/data/raw/iris.csv --version raw
python data_prep.py --src ga_resources/data/raw/iris.csv --version raw
python train.py --version raw
python inference.py --version raw
python train.py --version raw && python inference.py --version raw
python data_prep.py --src ga_resources/data/v1/data.csv --version v1
python train.py --version v1 && python inference.py --version v1
python data_prep.py --src ga_resources/data/v2/data.csv --version v2
python train.py --version v2 && python inference.py --version v2
git init && git checkout -b week_1
git add README.md requirements.txt config.py gcs_utils.py data_prep.py train.py inference.py .gitignore
git add README.md requirements.txt config.py gcs_utils.py data_prep.py train.py inference.py
git commit -m "Week 1: IRIS pipeline on Vertex AI + GCS"
git config --global user.email "ry9812262@gmail.com"
git config --global user.name "Ritik650"
git commit -m "Week 1: IRIS pipeline on Vertex AI + GCS"
git remote add origin https://github.com/Ritik650/23f2004634_MLOPS_WEEKLY_ASSIGNMENT.git
git push -u origin week_1
git push -u origin week_1
git push -u origin week_1
git push -u origin week_1
git push -u origin week_1
