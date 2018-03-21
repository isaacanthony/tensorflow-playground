import json
import os.path
import socket
import ssl
import urllib.error
import urllib.request

PWD  = 'kaggle/imaterialist'
SET  = 'validation' # 'train' # 'test'
data = json.load(open("{}/{}.json".format(PWD, SET)))

for image in data['images']:
    filepath = "{}/{}/{}".format(PWD, SET, image['image_id'])
    if not os.path.isfile(filepath):
        try:
            req = urllib.request.urlopen(image['url'][0], timeout=5)
            open(filepath, 'wb').write(req.read())
        except urllib.error.HTTPError:
            None
        except urllib.error.URLError:
            None
        except socket.timeout:
            None
        except ssl.CertificateError:
            None
