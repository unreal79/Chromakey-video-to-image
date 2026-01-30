set "INPUT=example/pic.png"
set "OUTPUT=example/output_pic"
py chromakey_video2png.py --find-best-image %INPUT% -o %OUTPUT%/find_best
py chromakey_video2png.py --image %INPUT% -o %OUTPUT% --bgr -c 1 -m 20 -k 3 -d 0 -b 1 --post-process --post-margin 270
