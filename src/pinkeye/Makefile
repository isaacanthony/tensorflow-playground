install:
	mkdir downloads
	wget https://raw.githubusercontent.com/ostrolucky/Bulk-Bing-Image-downloader/master/bbid.py -O bing.py
	wget https://raw.githubusercontent.com/deepanprabhu/duckduckgo-images-api/master/api.py -O duckduckgo_original.py
	head -n -2 duckduckgo_original.py > duckduckgo.py
	echo "\nimport sys;\nsearch(sys.argv[1]);" >> duckduckgo.py
	rm duckduckgo_original.py

download:
	python3 bing.py --search-string $(search) --output downloads/$(search)
	python duckduckgo.py $(search) | grep "Image http" | cut -d " " -f 2 | xargs wget -P downloads/$(search)

edit:
	find ./downloads -name '*' -exec mogrify -resize 299x299 -format jpg {} +

clean:
	rm -rf downloads
	rm -f bing.py
	rm -f duckduckgo.py
