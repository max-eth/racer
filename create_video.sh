
if [ "$#" -ne 1 ]; then
    echo "USAGE: create_video.sh outname.mp4"
    exit 1
fi
mkdir -p videos
rm videos/$1.mp4
ffmpeg -f image2 -framerate 60 -pattern_type sequence -i 'tmp/frames/%d.png' -c:v libx264 -pix_fmt yuv420p videos/$1.mp4
rm tmp/frames/*.png
