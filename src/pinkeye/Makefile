prep:
	touch keywords.txt
	mkdir downloads
	echo "id,class" > data.csv

download:
	googleimagesdownload -kf keywords.txt

install:
	sudo pip install google_images_download

tocsv:
	ruby tocsv.rb >> data.csv

clean:
	rm keywords.txt
	rm -r downloads
	rm data.csv
