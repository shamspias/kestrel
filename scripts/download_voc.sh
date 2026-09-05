#!/bin/bash
# download VOC07 trainval/test + VOC12 trainval; try pjreddie mirror then official host
set -u
mkdir -p data/voc && cd data/voc
get() { f=$1; shift; for u in "$@"; do
  if [ -s "$f" ] && tar -tf "$f" >/dev/null 2>&1; then echo "OK $f"; return 0; fi
  echo "GET $u"; curl -L --retry 5 --retry-delay 5 -C - -o "$f" "$u" && tar -tf "$f" >/dev/null 2>&1 && { echo "OK $f"; return 0; }
done; echo "FAIL $f"; return 1; }
get VOCtrainval_06-Nov-2007.tar https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
get VOCtest_06-Nov-2007.tar https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
get VOCtrainval_11-May-2012.tar https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
for f in *.tar; do echo "extract $f"; tar -xf "$f" || echo "EXTRACT FAIL $f"; done
ls VOCdevkit && du -sh VOCdevkit && echo DOWNLOAD_DONE
