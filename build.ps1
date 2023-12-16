jupyter nbconvert --to html  --template classic *.ipynb
python collate.py
mv *.html report/
cp *.svg report/
cp *.png report 