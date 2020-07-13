cd tmp/frames
rm out.mp4
ffmpeg -f image2 -framerate 60 -pattern_type sequence -i '%d.png' -c:v libx264 -pix_fmt yuv420p out.mp4
rm *.png
