# -*- coding: utf-8 -*-
import urllib2

# Swanson's 3rd edition brain map atlas
complete_url = "http://larrywswanson.com/wp-content/uploads/2015/03/Complete-Atlas-Level-"
map_url = "http://larrywswanson.com/wp-content/uploads/2015/03/Map-only-Atlas-Level-"
ext = ".pdf"

def fetch_atlas_level(level):
    fetch_website(complete_url + str(level).zfill(2) + ext)
    #fetch_website(map_url + str(level).zfill(2) + ext)
    
def fetch_website(url):
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)
    
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
    
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    
    print "\n"
    f.close()

# ---=== Main
print "Starting fetching script"

for i in range(1, 74):
    fetch_atlas_level(i)