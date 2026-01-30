set "INPUT=example/416530553804341248.mp4"
set "OUTPUT=example/output_vid"
py chromakey_video2png.py --find-best %INPUT% -o %OUTPUT%/find_best
py chromakey_video2png.py --video %INPUT% -o %OUTPUT% -f 1 --hsv -c 0 -m 20 -k 3 -d 0 -b 3 --post-process --post-margin 170
